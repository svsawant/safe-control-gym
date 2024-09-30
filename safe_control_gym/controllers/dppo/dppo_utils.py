'''DPPO utilities.'''

from collections import defaultdict
from copy import deepcopy
from importlib.metadata import distribution
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.math_and_models.distributions import Categorical, Normal
from safe_control_gym.math_and_models.neural_networks import MLP
from safe_control_gym.math_and_models.quantile_network import QuantileNetwork


class DPPOAgent:
    '''A DPPO class that encapsulates models, optimizers and update functions.'''

    critic_network: Type[nn.Module] = QuantileNetwork
    _alg_features = dict(recurrent=True)

    value_loss_energy = "sample_energy"
    value_loss_l1 = "quantile_l1"
    value_loss_huber = "quantile_huber"

    network_qrdqn = "qrdqn"

    networks = {network_qrdqn: QuantileNetwork}

    values_losses = {
        network_qrdqn: {
            value_loss_energy: QuantileNetwork.sample_energy_loss,
            value_loss_l1: QuantileNetwork.quantile_l1_loss,
            value_loss_huber: QuantileNetwork.quantile_huber_loss,
        }
    }

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=64,
                 use_clipped_value=False,
                 clip_param=0.2,
                 target_kl=0.01,
                 entropy_coef=0.01,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 opt_epochs=10,
                 mini_batch_size=64,
                 activation='tanh',
                 qrdqn_quantile_count: int = 200,
                 value_lambda: float = 0.95,
                 value_loss: str = value_loss_l1,
                 value_loss_kwargs=None,
                 value_measure: str = None,
                 value_measure_adaptation: Union[Tuple, None] = None,
                 value_measure_kwargs: Dict = {},
                 **kwargs
                 ):
        # Parameters.
        if value_loss_kwargs is None:
            value_loss_kwargs = {}
        self.obs_space = obs_space
        self.act_space = act_space
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.activation = activation
        self._critic_network_name = self.network_qrdqn  # current implementation is only done for qrdqn
        self.qrdqn_quantile_count = qrdqn_quantile_count
        self.value_lambda = value_lambda
        self.value_loss_name = value_loss
        self.value_loss_kwargs = value_loss_kwargs

        # Model.
        self.ac = MLPActorCritic(obs_space,
                                 act_space,
                                 hidden_dims=[hidden_dim] * 2,
                                 activation=self.activation)

        # Value loss
        value_loss_func = self.values_losses[self._critic_network_name][self.value_loss_name]
        self.value_loss = lambda *args, **kwargs: value_loss_func(self.ac.critic.v_net, *args, **kwargs)

        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    def to(self,
           device
           ):
        '''Puts agent to device.'''
        self.ac.to(device)

    def train(self):
        '''Sets training mode.'''
        self.ac.train()

    def eval(self):
        '''Sets evaluation mode.'''
        self.ac.eval()

    def state_dict(self):
        '''Snapshots agent state.'''
        return {
            'ac': self.ac.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict()
        }

    def load_state_dict(self,
                        state_dict
                        ):
        '''Restores agent state.'''
        self.ac.load_state_dict(state_dict['ac'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])

    def compute_policy_loss(self,
                            batch
                            ):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv = batch['obs'], batch['act'], batch['logp'], batch['adv']
        dist, logp = self.ac.actor(obs, act)
        # Policy.
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        # Entropy.
        entropy_loss = -dist.entropy().mean()
        # KL/trust region.
        approx_kl = (logp_old - logp).mean()
        return policy_loss, entropy_loss, approx_kl

    def compute_value_loss(self,
                           batch
                           ):
        '''Returns value loss(es) given batch of data.'''
        obs, ret, v_old = batch['obs'], batch['ret'], batch['v']
        v_cur_quant = self.ac.critic.forward(obs, distribution=True).squeeze()
        value_loss = self.value_loss(v_cur_quant, batch["value_target_quants"])

        # if self.use_clipped_value:
        #     v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
        #     v_loss = (v_cur - ret).pow(2)
        #     v_loss_clipped = (v_old_clipped - ret).pow(2)
        #     value_loss = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
        # else:
        #     value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def update(self,
               rollouts,
               device='cpu'
               ):
        '''Updates model parameters based on current training batch.'''
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        # assert if num_mini_batch is 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        for _ in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            for batch in rollouts.sampler(self.mini_batch_size, device):
                # Actor update.
                policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(batch)
                # Update only when no KL constraint or constraint is satisfied.
                if (self.target_kl <= 0) or (self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl):
                    self.actor_opt.zero_grad()
                    (policy_loss + self.entropy_coef * entropy_loss).backward()
                    self.actor_opt.step()

                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()

                p_loss_epoch += policy_loss.item()
                v_loss_epoch += value_loss.item()
                e_loss_epoch += entropy_loss.item()
                kl_epoch += approx_kl.item()
            results['policy_loss'].append(p_loss_epoch / num_mini_batch)
            results['value_loss'].append(v_loss_epoch / num_mini_batch)
            results['entropy_loss'].append(e_loss_epoch / num_mini_batch)
            results['approx_kl'].append(kl_epoch / num_mini_batch)
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results


class MLPActor(nn.Module):
    '''Actor MLP model.'''

    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dims,
                 activation,
                 discrete=False
                 ):
        super().__init__()
        self.pi_net = MLP(obs_dim, act_dim, hidden_dims, activation)
        # Construct output action distribution.
        self.discrete = discrete
        if discrete:
            self.dist_fn = lambda x: Categorical(logits=x)
        else:
            self.logstd = nn.Parameter(-0.5 * torch.ones(act_dim))
            self.dist_fn = lambda x: Normal(x, self.logstd.exp())

    def forward(self,
                obs,
                act=None
                ):
        dist = self.dist_fn(self.pi_net(obs))
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
        return dist, logp_a


class MLPCritic(nn.Module):
    '''Critic MLP model.'''

    def __init__(self,
                 obs_dim,
                 hidden_dims,
                 activation
                 ):
        super().__init__()
        self.v_net = QuantileNetwork(obs_dim, 1, hidden_dims, activation)

    def forward(self, obs, distribution: bool = False):
        # temp = self.v_net(obs).unsqueeze(0)
        return self.v_net(obs, distribution=distribution).unsqueeze(1)


class MLPActorCritic(nn.Module):
    '''Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dims=(64, 64),
                 activation='tanh'
                 ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
            discrete = False
        else:
            act_dim = act_space.n
            discrete = True
        # Policy.
        self.actor = MLPActor(obs_dim, act_dim, hidden_dims, activation, discrete)
        # Value function.
        self.critic = MLPCritic(obs_dim, hidden_dims, activation)

    def step(self,
             obs
             ):
        dist, _ = self.actor(obs)
        a = dist.sample()
        logp_a = dist.log_prob(a)
        v = self.critic.v_net(obs)
        v_quant = self.critic.v_net.last_quantiles.detach()
        return a.cpu().numpy(), v.cpu().numpy(), v_quant.cpu().numpy(), logp_a.cpu().numpy()

    def act(self,
            obs
            ):
        dist, _ = self.actor(obs)
        a = dist.mode()
        return a.cpu().numpy()


class DPPOBuffer(object):
    '''Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 max_length,
                 batch_size
                 ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        T, N = max_length, batch_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            'obs': {
                'vshape': (T, N, *obs_dim)
            },
            'act': {
                'vshape': (T, N, act_dim)
            },
            'rew': {
                'vshape': (T, N, 1)
            },
            'mask': {
                'vshape': (T, N, 1),
                'init': np.ones
            },
            'v': {
                'vshape': (T, N, 1)
            },
            'v_quant': {
                'vshape': (T, N, 200)
            },
            'logp': {
                'vshape': (T, N, 1)
            },
            'ret': {
                'vshape': (T, N, 1)
            },
            'adv': {
                'vshape': (T, N, 1)
            },
            'value_target_quants': {
                'vshape': (T, N, 200)
            },
            'terminal_v': {
                'vshape': (T, N, 1)
            },
            'terminal_v_quant': {
                'vshape': (T, N, 200)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.t = 0

    def push(self,
             batch
             ):
        '''Inserts transition step data (as dict) to storage.'''
        for k, v in batch.items():
            assert k in self.keys
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            # try:
            v_ = np.asarray(deepcopy(v), dtype=dtype).reshape(shape)
            # except:
            #     print(shape)
            #     print(v_)
            #     p()
            self.__dict__[k][self.t] = v_
        self.t = (self.t + 1) % self.max_length

    def get(self,
            device='cpu'
            ):
        '''Returns all data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self,
               indices
               ):
        '''Returns partial data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self,
                mini_batch_size,
                device='cpu',
                drop_last=True
                ):
        '''Makes sampler to loop through all data.'''
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch


def random_sample(indices,
                  batch_size,
                  drop_last=True
                  ):
    '''Returns index batches to iterate over.'''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(
        -1, batch_size)
    for batch in batches:
        yield batch
    if not drop_last:
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]


def compute_returns_and_advantages(rews,
                                   vals,
                                   val_quants,
                                   masks,
                                   terminal_vals=0,
                                   last_val=0,
                                   last_quant=0,
                                   gamma=0.99,
                                   use_gae=False,
                                   gae_lambda=0.95
                                   ):
    '''Useful for policy-gradient algorithms.'''
    T, N = rews.shape[:2]
    rets, advs, next_value_quants = np.zeros((T, N, 1)), np.zeros((T, N, 1)), np.zeros((T, N, 200))

    ret, ret_quant, adv = last_val, last_quant, np.zeros((N, 1))
    vals = np.concatenate([vals, last_val[np.newaxis, ...]], 0)

    # Compensate for time truncation.
    rews += gamma * terminal_vals

    # Cumulative discounted sums.
    for i in reversed(range(T)):
        ret = rews[i] + gamma * masks[i] * ret
        ret_quant = rews[i] + gamma * masks[i] * ret_quant
        preserved_value_quants = masks[i].astype(bool) * (np.random.rand(*ret_quant.shape) < 0.95)
        ret_quant_ = np.where(preserved_value_quants, ret_quant, val_quants[i])
        if not use_gae:
            adv = ret - vals[i]
        else:
            td_error = rews[i] + gamma * masks[i] * vals[i + 1] - vals[i]
            adv = adv * gae_lambda * gamma * masks[i] + td_error
        rets[i] = deepcopy(ret)
        advs[i] = deepcopy(adv)
        next_value_quants[i, :, :] = deepcopy(ret_quant)
        ret_quant = deepcopy(ret_quant_)
    return rets, advs, next_value_quants
