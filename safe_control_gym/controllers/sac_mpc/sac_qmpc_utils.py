"""SAC utilities."""

from collections import defaultdict, deque
from copy import deepcopy
import time

import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box

from safe_control_gym.controllers.ppo_mpc.rlmpc_utils import (
    update_initial_guess,
    MPCFunction,
)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.math_and_models.distributions import Normal
from safe_control_gym.math_and_models.neural_networks import MLP


class SAC_QMPC_Agent:
    """A SAC class that encapsulates models, optimizers and update functions."""

    def __init__(
        self,
        env_fun,
        gamma,
        model,
        hidden_dim=256,
        tau=0.005,
        init_temperature=0.2,
        use_entropy_tuning=False,
        target_entropy=None,
        exploration_init=-2.0,
        actor_lr=0.001,
        critic_lr=0.001,
        entropy_lr=0.001,
        activation="relu",
        actor_config=None,
        sigma_network=False,
        tanh_squash=False,
        update_freq=1,
        rollout_batch_size=10,
        train_batch_size=32,
        **kwargs,
    ):
        # Parameters.
        self.env = env_fun
        self.obs_space = env_fun.observation_space
        self.act_space = env_fun.action_space
        self.exploration_init = exploration_init
        self.gamma = gamma
        self.tau = tau
        self.use_entropy_tuning = use_entropy_tuning
        self.activation = activation
        self.rollout_batch_size = rollout_batch_size
        self.train_batch_size = train_batch_size
        self.update_freq = update_freq
        self.count = 0

        # Model.
        self.ac = MLPActorCritic(
            self.env,
            self.obs_space,
            self.act_space,
            gamma,
            model,
            hidden_dims=[hidden_dim] * 2,
            exploration_init=self.exploration_init,
            activation=self.activation,
            actor_config=actor_config,
            sigma_network=sigma_network,
            tanh_squash=tanh_squash,
            rollout_batch_size=self.rollout_batch_size,
            train_batch_size=self.train_batch_size,
            data_batch_size=self.train_batch_size * self.update_freq,
        )
        self.log_alpha = torch.tensor(np.log(init_temperature))

        # target networks
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Optimizers.
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), actor_lr)
        # self.critic_opt = torch.optim.Adam(
        #     list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()), critic_lr
        # )
        if self.use_entropy_tuning:
            self.log_alpha.requires_grad = True
            self.alpha_opt = torch.optim.Adam([self.log_alpha], entropy_lr)
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(self.act_space.shape).item()
            else:
                self.target_entropy = target_entropy
        else:
            self.alpha_opt = None

    @property
    def alpha(self):
        """Entropy-tuning parameter/temperature"""
        return self.log_alpha.exp()

    def to(self, device):
        """Puts agent to device."""
        self.ac.to(device)
        self.ac_targ.to(device)
        # self.log_alpha = self.log_alpha.to(device)
        self.log_alpha.data = self.log_alpha.data.to(device)

    def train(self):
        """Sets training mode."""
        self.ac.train()

    def eval(self):
        """Sets evaluation mode."""
        self.ac.eval()

    def reset(self, idx=None):
        """Reset function, especially needed for resetting MPC actor"""
        self.ac.reset(idx)

    def state_dict(self):
        """Snapshots agent state."""
        return {
            "ac": self.ac.state_dict(),
            "log_alpha": self.log_alpha,
            "ac_targ": self.ac_targ.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            # "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=True):
        """Restores agent state."""
        self.ac.load_state_dict(state_dict["ac"], strict=strict)
        self.log_alpha = self.log_alpha.to(next(self.ac.parameters()).device)
        self.ac_targ.load_state_dict(state_dict["ac_targ"], strict=strict)
        self.actor_opt.load_state_dict(state_dict["actor_opt"])
        # self.critic_opt.load_state_dict(state_dict["critic_opt"])
        self.alpha_opt.load_state_dict(state_dict["alpha_opt"])

    def compute_q_loss(
        self,
        batch,
        new_mpc_act,
        q_mpc,
        nabla_q_act,
        nabla_q_theta,
        nabla_q_act_theta,
        nabla_q_act_act,
        x0,
        optimal,
    ):
        """Returns q-value loss(es) given batch of data."""
        obs, act, next_obs = batch["obs"], batch["act"], batch["next_obs"]
        rew, mask, info = batch["rew"], batch["mask"], batch["info"]

        # MPC forward and backward for the next state
        (
            next_act,
            next_logp,
            next_mpc_act,
            _,
            _,
            next_q_mpc,
            _,
            nabla_q_act_next,
            nabla_q_theta_next,
            nabla_q_act_theta_next,
            nabla_q_act_act_next,
            next_optimal,
            x0,
        ) = self.ac.actor.forward_train(next_obs, info, update_info=True, x0=x0)

        Aq, bq = [], []
        for i in range(obs.shape[0]):
            if optimal[i, 0] > 0.9 and next_optimal[i, 0] > 0.9:
                da1 = act[i] - new_mpc_act[i].detach().cpu().numpy()
                da2 = (next_act[i] - next_mpc_act[i]).detach().cpu().numpy()
                nabla_q_aa = nabla_q_act_act[i].copy()
                nalba_q_aa = np.diag(np.diagonal(nabla_q_aa))
                nabla_q_aa_next = nabla_q_act_act_next[i].copy()
                nabla_q_aa_next = np.diag(np.diagonal(nabla_q_aa_next))

                # Q target
                q_targ = rew[i] + self.gamma * mask[i] * (
                    -next_q_mpc[i]
                    # - nabla_q_act_next[i] @ da2
                    # - nabla_q_theta_next[i] @ self.ac_targ.q1.weights[:-nu].detach().cpu().numpy()
                    # - da2[None, :] @ nabla_q_act_theta_next[i] @ self.ac_targ.q1.weights[:-nu].detach().cpu().numpy()
                    # - 0.5 * (sigma_sq * hessian_diag_next)[None, :] @ self.ac_targ.q1.weights[-nu:].detach().cpu().numpy()
                    # - 0.5 * da2[None, :] @ nabla_q_act_act_next[i] @ da2
                    - (self.alpha * next_logp[i]).detach().cpu().numpy()
                )

                # LSTDQ for linear Q-function approximation
                # A1 = (
                #     -nabla_q_theta[i][0, :] - da1 @ nabla_q_act_theta[i]
                #     + mask[i] * self.gamma * (nabla_q_theta_next[i][0, :] + da2 @ nabla_q_act_theta_next[i])
                # )
                A1 = (
                    -nabla_q_theta[i][0, :]
                    + mask[i] * self.gamma * nabla_q_theta_next[i][0, :]
                )
                A2 = -da1 @ nabla_q_act_theta[i] + mask[i] * self.gamma * (
                    da2 @ nabla_q_act_theta_next[i]
                )
                # A2 = - nabla_q_act[i] @ da1 + mask[i] * self.gamma * (nabla_q_act_next[i] @ da2)
                # A3 = -nabla_q_act[i][0, :] * da1 + mask[i] * self.gamma * (nabla_q_act_next[i][0, :] * da2)
                A4 = (
                    -0.5 * da1[None, :] @ nabla_q_aa * da1
                    + mask[i]
                    * self.gamma
                    * (0.5 * da2[None, :] @ nabla_q_aa_next * da2)
                )[0, :]
                A = np.concatenate([A1, A2, A4], axis=0)
                b = (
                    q_targ + q_mpc[i]
                )  # + 0.5 * da1[None, :] @ nabla_q_aa @ da1  #+ nabla_q_act[i] @ da1
                Aq.append(A)
                bq.append(b)

        self.ac.q1.Aq.extend(Aq)
        self.ac.q1.bq.extend(bq)

        if len(self.ac.q1.Aq) > 0:
            # Q value update
            AAq = np.asarray(self.ac.q1.Aq, dtype=np.float64)
            bbq = np.asarray(self.ac.q1.bq, dtype=np.float64).reshape(-1, 1)
            AAq = torch.FloatTensor(AAq)
            bbq = torch.FloatTensor(bbq)

            AtA = AAq.T @ AAq
            Atb = AAq.T @ bbq
            w = torch.linalg.solve(AtA + 1e-4 * torch.eye(AtA.shape[0]), Atb)
            self.ac.q1.weights.copy_(
                torch.as_tensor(w.squeeze(-1), dtype=torch.float32)
            )

            resid = AAq @ w.reshape(-1, 1) - bbq
            linear_q_loss = 0.5 * torch.mean(resid**2)
        else:
            # linear_q_loss = 0.0
            linear_q_loss = torch.tensor(0.0)
        return linear_q_loss

    def update(self, batch, batch_th):
        """Updates model parameters based on current training batch."""
        results = defaultdict(list)
        pip_sens_flag = True if self.count % self.update_freq == 0 else False

        # MPC forward and backward
        obs, obs_th, info = batch["obs"], batch_th["obs"], batch["info"]
        (
            new_act,
            logp,
            new_mpc_act,
            _,
            nabla_pi_theta,
            q_mpc,
            _,
            nabla_q_act,
            nabla_q_theta,
            nabla_q_act_theta,
            nabla_q_act_act,
            optimal,
            x0,
        ) = self.ac.actor.forward_train(obs, info, pi_sensitivities=pip_sens_flag)

        # critic update
        linear_q_loss = self.compute_q_loss(
            batch,
            new_mpc_act,
            q_mpc,
            nabla_q_act,
            nabla_q_theta,
            nabla_q_act_theta,
            nabla_q_act_act,
            x0,
            optimal,
        )
        results["critic_loss"] = linear_q_loss.item()

        # actor update
        if self.count % self.update_freq == 0:
            # compute policy loss and gradients
            q_m = torch.FloatTensor(q_mpc)
            nabla_q_a = torch.FloatTensor(np.array(nabla_q_act))
            nabla_q_t = torch.FloatTensor(np.array(nabla_q_theta))
            nabla_q_at = torch.FloatTensor(np.array(nabla_q_act_theta))
            nabla_q_aa = torch.FloatTensor(np.array(nabla_q_act_act))
            nabla_q_aa = torch.diag_embed(
                torch.diagonal(nabla_q_aa, dim1=-2, dim2=-1)
            )  # use only diagonal of hessian
            # nabla_q_aa_inv = torch.linalg.inv(nabla_q_aa + 1e-4 * torch.eye(nabla_q_aa.shape[-1]))
            # nabla_pi_t = -nabla_q_aa_inv @ nabla_q_at
            nabla_pi_t = nabla_pi_theta

            q_val = self.ac.q1(
                obs_th,
                new_act,
                new_mpc_act.detach(),
                q_m,
                nabla_q_a,
                nabla_q_t,
                nabla_q_at,
                nabla_q_aa,
            )
            policy_loss = self.alpha.detach() * logp - q_val
            policy_loss = torch.where(optimal > 0.9, policy_loss, torch.nan).nanmean()
            self.actor_opt.zero_grad()
            policy_loss.backward()

            # Passing the gradients through the mpc
            theta = self.ac.actor.get_theta_param(batch_th["obs"])
            theta_loss = (
                new_mpc_act.grad.unsqueeze(1) @ nabla_pi_t @ theta.unsqueeze(-1)
            ).sum()
            theta_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), max_norm=10.0)
            self.actor_opt.step()
            with torch.no_grad():
                self.ac.actor.q_param.clamp_(1e-5, 100.0)
                self.ac.actor.r_param.clamp_(1e-5, 100.0)
                self.ac.actor.qt_param.clamp_(1e-5, 100.0)
                self.ac.actor.model_param.clamp_(1e-5, 100.0)
                self.ac.actor.logstd.clamp_(
                    self.ac.actor.log_std_min, self.ac.actor.log_std_max
                )
            results["policy_loss"] = policy_loss.item()
            results["theta_loss"] = theta_loss.item()

            # compute entropy loss
            entropy_loss = torch.zeros(1)
            if self.use_entropy_tuning:
                entropy_loss = -(self.log_alpha * (logp + self.target_entropy).detach())
                entropy_loss = torch.where(
                    optimal > 0.9, entropy_loss, torch.nan
                ).nanmean()
                logp_loss = (logp + self.target_entropy).detach().mean()
                self.alpha_opt.zero_grad()
                entropy_loss.backward()
                self.alpha_opt.step()
            results["alpha"] = self.alpha.item()
            results["entropy_loss"] = entropy_loss.item()
            results["logp_loss"] = logp_loss.item()
        self.count += 1
        return results


# -----------------------------------------------------------------------------------
#                   Models
# -----------------------------------------------------------------------------------


class MLPActorCritic(nn.Module):
    """Model for the actor-critic agent.

    Attributes:
        actor (MLPActor): policy network.
        critic (MLPCritic): value network.
    """

    def __init__(
        self,
        env,
        obs_space,
        act_space,
        gamma,
        model,
        hidden_dims=(64, 64),
        exploration_init=-1.0,
        activation="tanh",
        actor_config=None,
        sigma_network=False,
        tanh_squash=False,
        rollout_batch_size=10,
        train_batch_size=32,
        data_batch_size=64,
    ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            raise Exception(
                "SAC-QMPC is currently only implemented for continuous action spaces"
            )
        # Policy.
        self.actor = MPCActor(
            env,
            obs_dim,
            act_dim,
            act_space,
            hidden_dims,
            activation,
            gamma,
            model,
            exploration_init,
            actor_config,
            sigma_network=sigma_network,
            tanh_squash=tanh_squash,
            rollout_batch_size=rollout_batch_size,
            train_batch_size=train_batch_size,
        )
        mpc_param = self.actor._build_mpc_param()
        # Q functions
        self.q1 = Critic(
            obs_dim,
            act_dim,
            mpc_param.shape[0],
            hidden_dims,
            activation,
            data_batch_size=data_batch_size,
        )

    def step(self, obs, info=None):
        a, soln_info = self.actor(obs, actor_info=info)
        return a.cpu().numpy(), soln_info

    def act(self, obs, info=None):
        a, _ = self.actor(obs, deterministic=True, actor_info=info)
        return a.cpu().numpy()

    def reset(self, idx=None):
        self.actor.reset(idx)


class Critic(nn.Module):
    """Linear Q function approximator."""

    def __init__(
        self, obs_dim, act_dim, input_dim, hidden_dims, activation, data_batch_size=32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.act_dim = act_dim
        self.Aq, self.bq = deque(maxlen=data_batch_size), deque(maxlen=data_batch_size)
        self.weights = nn.Parameter(
            torch.zeros(2 * input_dim + act_dim), requires_grad=False
        )

    def forward(
        self,
        obs,
        act,
        mpc_act,
        q_mpc,
        nabla_q_act,
        nabla_q_theta,
        nabla_q_act_theta,
        nabla_q_act_act,
    ):
        da = act - mpc_act
        q1 = (
            -q_mpc.unsqueeze(-1)
            - nabla_q_theta @ self.weights[: self.input_dim].unsqueeze(-1)
            - da.unsqueeze(1)
            @ nabla_q_act_theta
            @ self.weights[self.input_dim : 2 * self.input_dim].unsqueeze(-1)
            # - nabla_q_act * da.unsqueeze(1) @ self.weights[2*self.input_dim:2*self.input_dim+self.act_dim].unsqueeze(-1)
            # - (nabla_q_act * da.unsqueeze(-2)) @ self.weights[-self.act_dim:].unsqueeze(-1)
            # - 0.5 * self.weights[-1] * da.unsqueeze(1) @ nabla_q_act_act @ da.unsqueeze(-1)
            - 0.5
            * (da.unsqueeze(1) @ nabla_q_act_act * da.unsqueeze(1))
            @ self.weights[-self.act_dim :].unsqueeze(-1)
        ).squeeze(-1)
        return q1


class MPCActor(nn.Module):
    """Actor MPC model."""

    def __init__(
        self,
        env,
        obs_dim,
        act_dim,
        action_space,
        hidden_dims,
        activation,
        gamma,
        model,
        exploration_init,
        actor_config,
        sigma_network=False,
        tanh_squash=False,
        rollout_batch_size=10,
        train_batch_size=32,
    ):
        super().__init__()
        # mpc actor
        self.mpc = MPCPolicyFunction(
            env,
            gamma,
            model,
            **actor_config["mpc_config"],
            n_rollout_solver=rollout_batch_size,
            n_train_solver=train_batch_size,
        )

        # Parameters
        self.q_init = actor_config["q_mpc"]
        self.r_init = actor_config["r_mpc"]
        self.qt_init = actor_config["qt_mpc"]
        self.back_off_init = actor_config["back_off"]
        self.model_init = actor_config["model_param"]
        self.q_param = nn.Parameter(torch.tensor(self.q_init, dtype=torch.float32))
        self.r_param = nn.Parameter(torch.tensor(self.r_init, dtype=torch.float32))
        self.qt_param = nn.Parameter(torch.tensor(self.qt_init, dtype=torch.float32))
        self.model_param = nn.Parameter(
            torch.tensor(self.model_init, dtype=torch.float32)
        )
        # self.back_off_param = nn.Parameter(
        #     torch.tensor(self.back_off_init, dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "back_off_fixed", torch.tensor(self.back_off_init, dtype=torch.float32)
        # )

        # Construct output action distribution.
        self.sigma_network = sigma_network
        self.tanh_squash = tanh_squash
        self.log_std_min = -10.0
        self.log_std_max = 0.5
        if self.sigma_network:
            self.net = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation)
            self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)
        else:
            self.logstd = nn.Parameter(exploration_init * torch.ones(act_dim))
        self.dist_fn = lambda mu, log_std: Normal(mu, log_std.exp())

        # action rescaling (from cleanrl)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0, dtype=torch.float32
            ).flatten(),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0, dtype=torch.float32
            ).flatten(),
        )

    def forward(self, obs, deterministic=False, actor_info=None):
        theta = self.get_theta_param(obs)
        traj_param = self.get_references(actor_info)
        if obs.ndim > 1:
            act, info, _, _ = self.mpc.select_action_batch(
                obs, theta.numpy(), traj_param, actor_info
            )
        else:
            act, info, _, _ = self.mpc.select_action(obs, theta.numpy(), traj_param)
        act = torch.FloatTensor(np.array(act))
        # optimal_flag = torch.FloatTensor(np.array(optimal_flag))
        if self.tanh_squash:
            act = self.inverse_squashing(act)

        # action distribution
        if self.sigma_network:
            net_out = self.net(torch.FloatTensor(obs))
            log_std = self.log_std_layer(net_out)
        else:
            log_std = self.logstd
        dist = self.dist_fn(act, log_std)
        if deterministic:
            x_t = dist.mode()
        else:
            x_t = dist.rsample()
        if self.tanh_squash:
            # Squash the output
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            action = x_t
        return action, info

    def forward_train(
        self, obs, info, update_info=False, x0=None, pi_sensitivities=False
    ):
        theta = self.get_theta_param(obs)
        action, q_mpc, sensitivities_info, optimal_flag = (
            self.mpc.select_action_batch_train(
                obs,
                theta.detach().numpy(),
                info,
                update_info=update_info,
                x0_init=x0,
                pi_sensitivities=pi_sensitivities,
            )
        )
        action = torch.FloatTensor(action)
        # q_mpc = torch.FloatTensor(q_mpc)
        # x0 = torch.FloatTensor(sensitivities_info["x0"])
        nabla_pi_ref = torch.FloatTensor(np.array(sensitivities_info["nabla_pi_ref"]))
        nabla_pi_theta = torch.FloatTensor(
            np.array(sensitivities_info["nabla_pi_theta"])
        )
        optimal_flag = (
            torch.FloatTensor(optimal_flag).T
            if len(optimal_flag) > 0
            else torch.FloatTensor(np.ones((obs.shape[0], 1)))
        )
        action.requires_grad_()

        # action distribution
        if self.sigma_network:
            net_out = self.net(torch.FloatTensor(obs))
            log_std = self.log_std_layer(net_out)
        else:
            log_std = self.logstd
        if self.tanh_squash:
            # Inverse squashing
            z_t = self.inverse_squashing(action)
            dist = self.dist_fn(z_t, log_std)
        else:
            dist = self.dist_fn(action, log_std)
        x_t = dist.rsample()
        logp = dist.log_prob(x_t)

        # Squash the output
        if self.tanh_squash:
            y_t = torch.tanh(x_t)
            act = y_t * self.action_scale + self.action_bias
            logp -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6).sum(
                -1, keepdim=True
            )
            # logp -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(
            #     dim=1, keepdim=True
            # )
            # logp -= torch.log(self.action_scale).sum()
        else:
            act = x_t
        return (
            act,
            logp,
            action,
            nabla_pi_ref,
            nabla_pi_theta,
            q_mpc,
            sensitivities_info["nabla_q_ref"],
            sensitivities_info["nabla_q_act"],
            sensitivities_info["nabla_q_theta"],
            sensitivities_info["nabla_q_act_theta"],
            sensitivities_info["nabla_q_act_act"],
            optimal_flag,
            sensitivities_info["x0"],
        )

    def inverse_squashing(self, y):
        """Inverse of tanh squashing function."""
        y = (y - self.action_bias) / self.action_scale
        return torch.atanh(torch.clamp(y, -1.0 + 1e-3, 1.0 - 1e-3))

    def reset(self, idx=None):
        self.mpc.reset(idx)

    def _build_mpc_param(self):
        return torch.cat(
            [
                self.q_param,
                self.r_param,
                self.qt_param,
                # self.back_off_fixed,
                self.model_param,
            ],
            dim=0,
        )

    def get_theta_param(self, obs):
        theta = self._build_mpc_param()
        if obs.ndim > 1:
            return theta.unsqueeze(0).repeat(obs.shape[0], 1)
        return theta

    def get_references(self, info_batch):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        goal_states_batch = []
        for info in info_batch:
            traj_step = info["current_step"]
            traj_ref = info["x_ref"].T
            if self.mpc.env.TASK == Task.STABILIZATION:
                # Repeat goal state for horizon steps.
                goal_states = np.tile(
                    self.mpc.env.X_GOAL.reshape(-1, 1), (1, self.mpc.T + 1)
                )
            elif self.mpc.env.TASK == Task.TRAJ_TRACKING:
                # Slice trajectory for horizon steps, if not long enough, repeat last state.
                start = min(traj_step, traj_ref.shape[-1])
                end = min(traj_step + self.mpc.T + 1, traj_ref.shape[-1])
                remain = max(0, self.mpc.T + 1 - (end - start))
                goal_states = np.concatenate(
                    [traj_ref[:, start:end], np.tile(traj_ref[:, -1:], (1, remain))], -1
                )
            else:
                raise Exception("Reference for this mode is not implemented.")
            goal_states_batch.append(goal_states)
        return goal_states_batch  # list of (nx, T+1).


class MPCPolicyFunction(MPCFunction):
    def __init__(
        self,
        env_fun,
        gamma,
        model,
        horizon: int = 5,
        warmstart: bool = True,
        soft_constraints: bool = True,
        constraint_tol: float = 1e-6,
        additional_constraints: list = None,
        cs_workers: int = 1,
        jit: bool = False,
        jit_options: dict = None,
        n_rollout_solver: int = 1,
        n_train_solver: int = 1,
    ):
        super().__init__(
            env_fun,
            gamma,
            model,
            horizon=horizon,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            jit=jit,
            jit_options=jit_options,
        )
        self.cs_workers = cs_workers
        self.n_parallel_solver = n_rollout_solver
        self.n_train_solver = n_train_solver
        self.infos = [None] * self.n_parallel_solver

        # Parallel solvers
        self.pi_solvers, _, self.rkkt_norm_fns, _, _, _ = self.get_parallel_solver(
            self.n_parallel_solver, self.cs_workers
        )
        (
            self.pi_solvers_train,
            self.q_solver_train,
            self.rkkt_norm_fns_train,
            _,
            self.all_solvers_train,
            self.q_all_solvers_train,
        ) = self.get_parallel_solver(self.n_train_solver, self.cs_workers)

    def reset(self, idx=None):
        super().reset()
        if idx is not None:
            self.infos[idx] = None
        else:
            self.infos = [None] * self.n_parallel_solver

    def setup_optimizer(self):
        super().setup_optimizer()
        qz = self.solver_dict["qz"]
        fixed_param = self.solver_dict["fixed_param"]
        ref_param = self.solver_dict["ref_param"]
        theta = self.solver_dict["theta_param"]
        jit_opts = self.solver_dict["jit_options"]
        dQda = self.q_sensitivity_dict["dQda"]
        dQdtheta = self.q_sensitivity_dict["dQdtheta"]
        dQdaa = self.q_sensitivity_dict["dqdaa"]
        dQdaP = self.q_sensitivity_dict["dqdaP"]
        start_time = time.time()

        # Jitting the q sensitivity function for training
        all_fn = cs.Function(
            "all_fn",
            [qz, fixed_param, ref_param, theta],
            [dQda, dQdtheta, dQdaa, dQdaP],
            jit_opts,
        )
        # all_fn.save("all_fn.casadi")
        print(
            f"[MPC Setup] JIT compilation time: {time.time() - start_time:.3f} seconds."
        )
        self.q_sensitivity_dict["all_fn"] = all_fn

    def select_action_batch(self, obs_batch, theta, traj_ref, agent_info):
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        con_lbg = self.solver_dict["lower_bound"]
        con_ubg = self.solver_dict["upper_bound"]
        opt_vars_fn = self.solver_dict["opt_vars_fn"]
        xus_fn = self.solver_dict["xus_fn"]

        # eval_data_batch = []
        x0, fixed_p, ref_p = [], [], []
        # ref_param = goal_states.T.reshape(-1, 1).repeat(obs_batch.shape[0], 1)
        lbg = con_lbg.full().repeat(obs_batch.shape[0], 1)
        ubg = con_ubg.full().repeat(obs_batch.shape[0], 1)
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        for i, obs in enumerate(obs_batch):
            fixed_param = np.zeros((self.model.nx + self.model.nu))
            fixed_param[: self.model.nx] = obs[: self.model.nx]
            ref_param = traj_ref[i].T.reshape(-1, 1)[:, 0]
            opt_vars_init = np.zeros(self.solver_dict["opt_vars"].shape)

            # shift previous solutions by 1 step based on last soln
            info = agent_info[i]["soln_info"]
            if info is not None:
                opt_vars_init = info["opt_var"]
            elif self.infos[i] is not None:
                opt_vars_init = self.infos[i]["opt_var"]
            else:
                opt_vars_init = np.zeros_like(opt_vars_init)
            x_prev, u_prev, sigma_prev, sigma_u0_prev = xus_fn(opt_vars_init)
            x_prev, u_prev, sigma_prev, sigma_u0_prev = (
                x_prev.full(),
                u_prev.full(),
                sigma_prev.full(),
                sigma_u0_prev.full(),
            )
            opt_vars_init = update_initial_guess(
                x_prev, u_prev, sigma_prev, sigma_u0_prev, opt_vars_fn
            )

            x0.append(opt_vars_init[:, 0])
            fixed_p.append(fixed_param)
            ref_p.append(ref_param)
        x0, fixed_p, ref_p = np.array(x0).T, np.array(fixed_p).T, np.array(ref_p).T
        p = np.concatenate((fixed_p, ref_p, theta.T), axis=0)

        # Forward pass through solver
        soln_batch = self.pi_solvers(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z = cs.vertcat(soln_batch["x"], soln_batch["lam_g"])
        rkkt_norm_batch = self.rkkt_norm_fns(z, fixed_p, ref_p, theta.T)
        optimal_batch = rkkt_norm_batch.full() < 1e-3

        # Post-processing the solution
        action_batch, results_dict_batch, info_batch = [], [], []
        for i, obs in enumerate(obs_batch):
            opt_vars = soln_batch["x"].full()[:, i]
            x_val, u_val, sigma_val, sigma_u0_val = xus_fn(opt_vars)
            x_prev = x_val.full()
            u_prev = u_val.full()
            sigma_prev = sigma_val.full()
            sigma_u0_prev = sigma_u0_val.full()
            results_dict = {
                "horizon_states": deepcopy(x_prev),
                "horizon_inputs": deepcopy(u_prev),
                "horizon_slacks": deepcopy(sigma_prev),
                "horizon_u0_slacks": deepcopy(sigma_u0_prev),
                "goal_states": deepcopy(ref_p[:, i]),
            }
            # results_dict['t_wall'].append(opti.stats()['t_wall_total'])

            # Take the first action from the solved action sequence.
            if u_prev.ndim > 1:
                action = u_prev[:, 0]
            else:
                action = np.array([u_prev[0]])

            # additional info
            info = {
                "success": optimal_batch[0, i],
                "opt_var": opt_vars,
                "fixed_param": deepcopy(fixed_p[:, i]),
                "ref_param": deepcopy(ref_p[:, i]),
                "theta_param": deepcopy(theta[i, :]),
                "traj_step": deepcopy(agent_info[i]["current_step"]),
                "x_ref": deepcopy(agent_info[i]["x_ref"]),
            }

            # result batch
            action_batch.append(action)
            results_dict_batch.append(results_dict)
            info_batch.append(info)
        self.infos = deepcopy(info_batch)
        return action_batch, info_batch, results_dict_batch, optimal_batch

    def select_action_batch_train(
        self,
        obs_batch,
        theta,
        info_batch,
        update_info=False,
        x0_init=None,
        pi_sensitivities=False,
    ):
        con_lbg = self.solver_dict["lower_bound"]
        con_ubg = self.solver_dict["upper_bound"]
        qcon_lbg = self.solver_dict["qlower_bound"]
        qcon_ubg = self.solver_dict["qupper_bound"]
        opt_act_fn = self.solver_dict["opt_act_fn"]
        opt_vars_fn = self.solver_dict["opt_vars_fn"]
        xus_fn = self.solver_dict["xus_fn"]
        npl = self.solver_dict["theta_param"].shape[0]
        nx, nu = self.model.nx, self.model.nu

        x0, fixed_p, ref_p = [], [], []
        lbg = con_lbg.full().repeat(obs_batch.shape[0], 1)
        ubg = con_ubg.full().repeat(obs_batch.shape[0], 1)
        qlbg = qcon_lbg.full().repeat(obs_batch.shape[0], 1)
        qubg = qcon_ubg.full().repeat(obs_batch.shape[0], 1)
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        for i, obs in enumerate(obs_batch):
            info = info_batch[i]
            opt_vars_init = info["opt_var"]
            if x0_init is not None:
                opt_vars_init = x0_init[:, i]
            fixed_param = np.zeros((self.model.nx + self.model.nu))
            fixed_param[: self.model.nx] = obs[: self.model.nx]
            ref_param = info["ref_param"]
            if update_info:
                # update the optimization variable init
                x_prev, u_prev, sigma_prev, sigma_u0_prev = xus_fn(opt_vars_init)
                x_prev, u_prev, sigma_prev, sigma_u0_prev = (
                    x_prev.full(),
                    u_prev.full(),
                    sigma_prev.full(),
                    sigma_u0_prev.full(),
                )
                opt_vars_init = update_initial_guess(
                    x_prev, u_prev, sigma_prev, sigma_u0_prev, opt_vars_fn
                )[:, 0]
                # update the reference parameter
                ref_param = self.get_references(
                    info["traj_step"] + 1, info["x_ref"].T
                ).T.reshape(-1, 1)[:, 0]

            x0.append(opt_vars_init)
            fixed_p.append(fixed_param)
            ref_p.append(ref_param)
        x0, fixed_p, ref_p = np.array(x0).T, np.array(fixed_p).T, np.array(ref_p).T
        p = np.concatenate((fixed_p, ref_p, theta.T), axis=0)

        # Forward pass through solver
        soln_batch = self.pi_solvers_train(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z = cs.vertcat(soln_batch["x"], soln_batch["lam_g"])
        action_batch = opt_act_fn(soln_batch["x"]).full().T

        # Forward pass for Q solver
        qfixed_p = fixed_p.copy()
        qfixed_p[self.model.nx :, :] = action_batch.T
        qx0 = soln_batch["x"].full()
        qp = np.concatenate((qfixed_p, ref_p, theta.T), axis=0)
        qsoln_batch = self.q_solver_train(x0=qx0, p=qp, lbg=qlbg, ubg=qubg)
        qz = cs.vertcat(qsoln_batch["x"], qsoln_batch["lam_g"])
        q_mpc_batch = qsoln_batch["f"].full().T

        # Post-processing the solution
        nabla_pi_ref_batch, nabla_pi_theta_batch, optimal_batch = [], [], []
        nabla_q_ref_batch, nabla_q_theta_batch, nabla_q_act_theta_batch = [], [], []
        nabla_q_act_batch, nabla_q_act_act_batch = [], []
        dqa_cs, dqt_cs, dqdaa_cs, dqdat_cs = self.q_all_solvers_train(
            qz, qfixed_p, ref_p, theta.T
        )
        dqa_cs = dqa_cs.full()
        dqt_cs = dqt_cs.full()
        dqdaa_cs = dqdaa_cs.full()
        dqdat_cs = dqdat_cs.full()
        for i in range(obs_batch.shape[0]):
            # nabla_q_ref_batch.append(dq_cs[:, npl * i : npl * i + self.model.nx])
            nabla_q_act_batch.append(
                dqa_cs[
                    :,
                    nu * i : nu * (i + 1),
                ]
            )
            nabla_q_act_act_batch.append(
                dqdaa_cs[
                    :,
                    nu * i : nu * (i + 1),
                ]
            )
            nabla_q_theta_batch.append(
                dqt_cs[
                    :,
                    npl * i : npl * (i + 1),
                ]
            )
            nabla_q_act_theta_batch.append(
                dqdat_cs[
                    :,
                    npl * i : npl * (i + 1),
                ]
            )
        if pi_sensitivities:
            # dpi_cs = dpi_fn_train(optimal_batch, z, fixed_p, ref_p, theta.T).full()
            rkkt_norm_batch, dpi_cs = self.all_solvers_train(z, fixed_p, ref_p, theta.T)
            optimal_batch = rkkt_norm_batch.full() < 1e-3
            for i in range(obs_batch.shape[0]):
                nabla_pi_theta_batch.append(
                    int(optimal_batch[0, i])
                    * dpi_cs[
                        :,
                        npl * i : npl * (i + 1),
                    ]
                )
        else:
            rkkt_norm_batch = self.rkkt_norm_fns_train(z, fixed_p, ref_p, theta.T)
            optimal_batch = rkkt_norm_batch.full() < 1e-3
        sensitivities_info_batch = {
            "x0": soln_batch["x"].full(),
            "nabla_pi_ref": nabla_pi_ref_batch,
            "nabla_pi_theta": nabla_pi_theta_batch,
            "nabla_q_ref": nabla_q_ref_batch,
            "nabla_q_act": nabla_q_act_batch,
            "nabla_q_theta": nabla_q_theta_batch,
            "nabla_q_act_theta": nabla_q_act_theta_batch,
            "nabla_q_act_act": nabla_q_act_act_batch,
        }
        return (
            action_batch,
            q_mpc_batch,
            sensitivities_info_batch,
            optimal_batch,
        )

    def get_parallel_solver(self, n_solvers, cs_workers):
        n_workers = min(n_solvers, cs_workers)
        pi_solvers = self.solver_dict["solver"].map(n_solvers, "thread", n_workers)
        q_solver = self.solver_dict["qsolver"].map(n_solvers, "thread", n_workers)
        rkkt_norm_fns = self.pi_sensitivity_dict["rkkt_norm_fn"].map(
            n_solvers, "thread", n_workers
        )
        dpi_fns = self.pi_sensitivity_dict["dpi_fn"].map(n_solvers, "thread", n_workers)
        all_fns = self.pi_sensitivity_dict["all_fn"].map(n_solvers, "thread", n_workers)
        q_all_fns = self.q_sensitivity_dict["all_fn"].map(
            n_solvers, "thread", n_workers
        )
        return pi_solvers, q_solver, rkkt_norm_fns, dpi_fns, all_fns, q_all_fns


class SACBuffer(object):
    """Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    """

    def __init__(self, obs_space, act_space, max_size, batch_size):
        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size
        N = max_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            "obs": {"vshape": (N, *obs_dim)},
            "next_obs": {"vshape": (N, *obs_dim)},
            "act": {"vshape": (N, act_dim)},
            "rew": {"vshape": (N, 1)},
            "mask": {"vshape": (N, 1), "init": np.ones},
            "info": {"vshape": (N,), "dtype": object, "init": np.empty},
            # 'optimal': {'vshape': (N, 1)},
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        """Allocates space for containers."""
        for k, info in self.scheme.items():
            assert "vshape" in info, f"Scheme must define vshape for {k}"
            # self.__dict__[k] = deque([], maxlen=self.max_size)
            # if k in ['info', 'results_dict']:
            #     self.__dict__[k] = deque([], maxlen=info['vshape'])
            # else:
            vshape = info["vshape"]
            dtype = info.get("dtype", np.float32)
            init = info.get("init", np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        """Returns current size of the buffer."""
        return self.buffer_size

    def state_dict(self):
        """Returns a snapshot of current buffer."""
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self, state):
        """Restores buffer from previous state."""
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self, batch):
        """Inserts transition step data (as dict) to storage."""
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            # assert k in self.keys
            if k == "info":
                # v should be list-like length n (e.g., list of dicts)
                assert len(v) == n, "info length mismatch"
                if self.pos + n <= self.max_size:
                    self.info[self.pos : self.pos + n] = v
                else:
                    remain_n = self.pos + n - self.max_size
                    self.info[self.pos : self.max_size] = v[:-remain_n]
                    self.info[:remain_n] = v[-remain_n:]
                continue

            # if k not in ['info']:
            shape = self.scheme[k]["vshape"][1:]
            dtype = self.scheme[k].get("dtype", np.float32)
            v = np.asarray(v, dtype=dtype).reshape((n,) + shape)
            if self.pos + n <= self.max_size:
                self.__dict__[k][self.pos : self.pos + n] = v
            else:
                # wrap around
                remain_n = self.pos + n - self.max_size
                self.__dict__[k][self.pos : self.max_size] = v[:-remain_n]
                self.__dict__[k][:remain_n] = v[-remain_n:]
            # else:
            #     self.__dict__[k].extend(v)
        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)
        self.pos = (self.pos + n) % self.max_size

    def sample(self, batch_size=None, device=None):
        """Returns data batch."""
        if not batch_size:
            batch_size = self.batch_size

        indices = np.random.randint(0, len(self), size=batch_size)
        batch, batch_th = {}, {}
        for k, info in self.scheme.items():
            # if k in ['info']:
            #     batch[k] = [self.__dict__[k][i] for i in indices]
            if k == "info":
                batch[k] = self.info[indices].tolist()
            else:
                shape = info["vshape"][1:]
                # batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
                # batch_th[k] = self.__dict__[k].reshape(-1, *shape)[indices]
                v = self.__dict__[k].reshape(-1, *shape)[indices]
                batch[k] = v
                if device is None:
                    batch_th[k] = torch.as_tensor(v)
                else:
                    batch_th[k] = torch.as_tensor(v, device=device)
            # data = list(self.__dict__[k])
            # batch[k] = [data[i] for i in indices]
        # for k, v in batch.items():
        #     if k not in ['info', 'results_dict']:
        #         batch_th[k] = torch.as_tensor(np.array(v), dtype=torch.float32, device=device)
        return batch, batch_th


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


def soft_update(source, target, tau):
    """Synchronizes target networks with exponential moving average."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(source, target):
    """Synchronizes target networks by copying over parameters directly."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
