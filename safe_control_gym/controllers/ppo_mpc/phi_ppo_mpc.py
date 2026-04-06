"""Proximal Policy Optimisation with Phi-MPC (PhiPPO_MPC).

Subclasses PPO_MPC_Agent and PPO_MPC to replace the symbolic model-based MPC
with a Phi-predictor-based QP, while inheriting the PPO training infrastructure.

=== WHAT IS OVERRIDDEN ===

PhiPPO_MPC_Agent(PPO_MPC_Agent):
  __init__             — builds PhiMLPActorCritic instead of MLPActorCritic.
                         Does NOT call super().__init__() because that creates an
                         MPCPolicyFunction with symbolic dynamics which we don't need.
  update()             — identical to PPO_MPC_Agent.update() except for selective
                         cost_param clamping: only cost weights clamped, not Phi entries.
  compute_value_loss() — critic trains on normalised z_past instead of obs.

PhiPPO_MPC(PPO_MPC):
  __init__       — wraps envs in PhiEnv, loads Phi, creates PhiPPO_MPC_Agent.
                   Calls BaseController.__init__ directly (not PPO_MPC.__init__)
                   to avoid building a throwaway symbolic MPCPolicyFunction.
  train_step()   — wires z_past from PhiEnv into the actor QP and critic.
  run()          — passes z_past from PhiEnv to the actor at each eval step.
  save()/load()  — extend PPO_MPC to also persist the z_past normaliser.
  log_step()     — prints phi_norm instead of the full mpc_param vector.
  + helper methods: _expected_eval_ep_length(), _is_full_length_eval().

=== WHAT IS INHERITED ===

PPO_MPC_Agent: to, train, eval, reset, state_dict, load_state_dict,
               compute_policy_loss

PPO_MPC:       reset, close
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")   # headless — no display required during training
import matplotlib.pyplot as plt
import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.ppo_mpc.phi_env import PhiEnv
from safe_control_gym.controllers.ppo_mpc.phi_mpc_utils import PhiMLPActorCritic
from safe_control_gym.controllers.ppo_mpc.ppo_mpc import PPO_MPC
from safe_control_gym.controllers.ppo_mpc.ppo_mpc_utils import (
    PPO_MPC_Agent,
    PPOBuffer,
    compute_returns_and_advantages,
)
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (
    RecordEpisodeStatistics,
    VecRecordEpisodeStatistics,
)
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.math_and_models.normalization import (
    BaseNormalizer,
    MeanStdNormalizer,
    RewardStdNormalizer,
)
from safe_control_gym.utils.logging import ExperimentLogger


class PhiPPO_MPC_Agent(PPO_MPC_Agent):
    """PPO agent whose policy is a Phi-MPC QP.

    Overrides __init__, update(), and compute_value_loss().
    All other methods (to, train, eval, reset, state_dict, load_state_dict,
    compute_policy_loss) are inherited from PPO_MPC_Agent.
    """

    def __init__(
        self,
        env_fun,
        gamma: float,
        phi_past_init: np.ndarray,
        phi_future_init: np.ndarray,
        mask_future: np.ndarray,
        hidden_dim: int,
        activation: str,
        actor_config: dict,
        use_clipped_value: bool,
        clip_param: float,
        target_kl: float,
        entropy_coef: float,
        exploration_init: float,
        actor_lr: float,
        critic_lr: float,
        opt_epochs: int,
        mini_batch_size: int,
        phi_lr_scale: float,
        grad_clip: float,
        z_past_dim: int,
        critic_obs_dim: int = None,
        **kwargs,
    ):
        self.env = env_fun
        self.obs_space = env_fun.observation_space
        self.act_space = env_fun.action_space
        self.use_clipped_value = use_clipped_value
        self.clip_param = clip_param
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.exploration_init = exploration_init
        self.opt_epochs = opt_epochs
        self.mini_batch_size = mini_batch_size
        self.activation = activation
        self.phi_lr_scale = phi_lr_scale
        self.grad_clip = grad_clip


        # Build actor-critic with PhiMPCActor 
        self.ac = PhiMLPActorCritic(
            env_fun,
            self.obs_space,
            self.act_space,
            gamma,
            phi_past_init,
            phi_future_init,
            mask_future,
            hidden_dims=(hidden_dim, hidden_dim),
            exploration_init=exploration_init,
            activation=activation,
            actor_config=actor_config,
            critic_obs_dim=critic_obs_dim if critic_obs_dim is not None else z_past_dim,
        )

        # Facilitate possible separate learning rate for MPC cost params and Phi 
        non_phi_params = [self.ac.actor.cost_param, self.ac.actor.logstd]
        non_phi_params += list(self.ac.actor.param_net.parameters())
        self.actor_opt = torch.optim.Adam([
            {"params": non_phi_params, "lr": actor_lr},
            {"params": [self.ac.actor.phi_param], "lr": actor_lr * self.phi_lr_scale},
        ])
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), critic_lr)

    def update(self, rollouts, device="cpu"):
        """PPO update — identical to PPO_MPC_Agent.update() except clamping."""
        
        results = defaultdict(list)
        
        # Figure out how many minibatches this rollout batch will be split into
        num_mini_batch = (
            rollouts.max_length * rollouts.batch_size // self.mini_batch_size
        )
        assert num_mini_batch != 0, "num_mini_batch is 0"

        kkt_norms_all = []

        # For *amount of opt_epochs*:
        for _ in range(self.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            theta_loss_epoch, ref_loss_epoch = 0, 0
            valid_frac_epoch = 0

            # Loop through the rollout batch in minibatches
            for batch, batch_th in rollouts.sampler(self.mini_batch_size, device):
                # Figure out how many of the rollout samples came from good (not broken) QP solves
                valid_frac_epoch += (batch_th["optimal"] > 0.9).float().mean().item()
                
                # Find policy loss
                (
                    policy_loss,
                    entropy_loss,
                    approx_kl,
                    action_th,
                    nabla_pi_ref,
                    nabla_pi_theta,
                ) = self.compute_policy_loss(batch, batch_th)

                # Collect KKT norms stored by select_action_batch_train
                if hasattr(self.ac.actor.mpc, '_last_kkt_norms'):
                    kkt_norms_all.append(self.ac.actor.mpc._last_kkt_norms.flatten())

                # Only do the update if policy has not moved too far
                if (self.target_kl <= 0) or (
                    self.target_kl > 0 and approx_kl <= 1.5 * self.target_kl
                ):
                    # Clear old actor gradients
                    self.actor_opt.zero_grad()
                    (policy_loss + self.entropy_coef * entropy_loss).backward()

                    # KKT sensitivity: propagate RL gradient into mpc_param.
                    # theta_loss = (dL/du*_0) @ (du*_0/dtheta) @ theta
                    theta = self.ac.actor.get_theta_param(batch_th["obs"])
                    theta_loss = (
                        action_th.grad.unsqueeze(1)
                        @ nabla_pi_theta
                        @ theta.unsqueeze(2)
                    )
                    (theta_loss.sum()).backward()

                    # Update parameters
                    self.actor_opt.step()

                    # Per-parameter clamping (Phi is unconstrained)
                    with torch.no_grad():
                        cp = self.ac.actor.cost_param.data
                        n_y = self.ac.actor.mpc.n_y
                        n_u = self.ac.actor.mpc.n_u
                        # Qy, R, tw: [1e-5, 100]
                        cp[:n_y + n_u + 1].clamp_(1e-5, 100.0)
                        # ws (w_sigma): [1e-2, 1e4]
                        cp[n_y + n_u + 1].clamp_(1e-2, 1e4)
                        # tm (theta_max): [0.05, inf) — no upper bound, RL tunes freely
                        cp[n_y + n_u + 2].clamp_(min=0.05)

                    p_loss_epoch += policy_loss.item()
                    e_loss_epoch += entropy_loss.item()
                    kl_epoch += approx_kl.item()
                    theta_loss_epoch += theta_loss.sum().item()

                # Update critic
                value_loss = self.compute_value_loss(batch_th)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
                v_loss_epoch += value_loss.item()

            results["policy_loss"].append(p_loss_epoch / num_mini_batch)
            results["value_loss"].append(v_loss_epoch / num_mini_batch)
            results["entropy_loss"].append(e_loss_epoch / num_mini_batch)
            results["approx_kl"].append(kl_epoch / num_mini_batch)
            results["theta_loss"].append(theta_loss_epoch / num_mini_batch)
            results["ref_loss"].append(ref_loss_epoch / num_mini_batch)
            results["valid_frac"].append(valid_frac_epoch / num_mini_batch)

        results = {k: sum(v) / len(v) for k, v in results.items()}

        # Aggregate KKT norms across all mini-batches/epochs.
        if kkt_norms_all:
            kkt_all = np.concatenate(kkt_norms_all)
            results["kkt_norm_mean"] = float(kkt_all.mean())
            results["kkt_norm_max"] = float(kkt_all.max())
            results["kkt_norm_min"] = float(kkt_all.min())

        return results

    def compute_value_loss(self, batch_th):
        """Critic trains on normalised z_past (I/O history)."""
        z_past, ret, v_old = batch_th["z_past"], batch_th["ret"], batch_th["v"]
        v_cur = self.ac.critic(z_past)
        if self.use_clipped_value:
            v_old_clipped = v_old + (v_cur - v_old).clamp(-self.clip_param, self.clip_param)
            v_loss = (v_cur - ret).pow(2)
            v_loss_clipped = (v_old_clipped - ret).pow(2)
            return 0.5 * torch.max(v_loss, v_loss_clipped).mean()
        return 0.5 * (v_cur - ret).pow(2).mean()


class PhiPPO_MPC(PPO_MPC):
    """Proximal Policy Optimisation with Phi-MPC policy.

    Overrides __init__, reset_before_run(), select_action(), train_step(), run(),
    save(), load(), and log_step().
    Also adds helper methods: _expected_eval_ep_length(), _is_full_length_eval().
    Everything else (reset, close) is inherited from PPO_MPC.
    """
    
    def __init__(
        self,
        env_func,
        phi_path: str,
        phi_bias_path: str | None,
        training: bool = True,
        checkpoint_path: str = "model_latest.pt",
        output_dir: str = "temp",
        use_gpu: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        # Call BaseController.__init__ (not PPO_MPC.__init__) to avoid creating
        # a throwaway PPO_MPC_Agent with an MPCPolicyFunction 
        BaseController.__init__(
            self, env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs
        )

        # Wrap env_func so every instantiation returns a PhiEnv.
        # PhiEnv filters observations to y_indices, maintains the I/O history
        # buffer, and exposes z_past and get_predictor().
        mpc_cfg = self.actor_config["mpc_config"]
        t_ini = int(mpc_cfg["t_ini"])
        y_indices = list(mpc_cfg["y_indices"])
        horizon = int(mpc_cfg["horizon"])
        
        _raw_env_func = env_func
        
        # Redefine env_func such that it instantiates our PhiWrapper environment
        env_func = lambda *args, **kwargs: PhiEnv(
            _raw_env_func(*args, **kwargs), y_indices, t_ini, phi_path, horizon,
            raw_env_func=_raw_env_func,
        )

        # Fetch linear multi-step predictor and deduce dimensions
        self.env = env_func()
        phi_past_init, phi_future_init, mask_future = self.env.get_predictor()
        z_past_dim = self.env.z_past_dim
        self.n_y = len(y_indices)
        self.n_u = int(np.asarray(self.env.action_space.shape).prod())
        self._phi_y_indices = y_indices
        self._phi_t_ini = t_ini
        self._phi_u_eq = self.env.u_eq.copy()

        # Environment setup
        if self.training:
            self.venv = make_vec_envs(
                env_func, None, self.rollout_batch_size, self.num_workers, seed
            )
            
            # Wrap the rollout envs in a statistics tracking layer
            self.venv = VecRecordEpisodeStatistics(self.venv, self.deque_size)
            
            self.eval_venv = env_func(randomized_init=False, seed=seed * 111)
            self.eval_venv = RecordEpisodeStatistics(self.eval_venv, self.deque_size)
        else:
            # Eval only
            self.env = RecordEpisodeStatistics(self.env)


        # Build the Phi-MPC agent
        self.agent = PhiPPO_MPC_Agent(
            self.env,
            self.gamma,
            phi_past_init,
            phi_future_init,
            mask_future,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            actor_config=self.actor_config,
            use_clipped_value=self.use_clipped_value,
            clip_param=self.clip_param,
            target_kl=self.target_kl,
            entropy_coef=self.entropy_coef,
            exploration_init=self.exploration_init,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            opt_epochs=self.opt_epochs,
            mini_batch_size=self.mini_batch_size,
            phi_lr_scale=self.phi_lr_scale,
            grad_clip=self.grad_clip,
            z_past_dim=z_past_dim,
            critic_obs_dim=z_past_dim,
        )
        self.agent.to(self.device)

        # Save initial parameters for experiment summary
        self._initial_cost_param = self.agent.ac.actor.cost_param.detach().cpu().numpy().copy()
        self._initial_phi_norm = float(np.linalg.norm(
            self.agent.ac.actor.phi_param.detach().cpu().numpy()
        ))
        self._best_eval_rmse = float("inf")
        self._best_eval_rmse_full_length = float("inf")
        self._external_eval_env = None
        self._external_u_hist = None
        self._external_y_hist = None
        self._external_eval_needs_history_update = False

        # Normalizers (currently only normalizing input to critic)
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(
                shape=(
                    self.venv.observation_space.shape
                    if self.training
                    else self.env.observation_space.shape
                ),
                clip=self.clip_obs,
                epsilon=1e-8,
            )
        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(
                gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8
            )

        self._z_past_dim = z_past_dim
        self.z_past_normalizer = MeanStdNormalizer(
            shape=(self._z_past_dim,),
            clip=10.0,
            epsilon=1e-8,
        )

        # Logger
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(
            output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard
        )

    def _extract_phi_obs(self, obs):
        """Return the Phi controller observation [x, z, theta]."""
        obs_arr = np.asarray(obs, dtype=np.float64)
        if obs_arr.shape[0] == self.n_y:
            return obs_arr.astype(np.float32)
        return obs_arr[self._phi_y_indices].astype(np.float32)

    def _bind_external_eval_env(self, obs, env):
        """Initialise controller-side I/O history for BaseExperiment evaluation."""
        self._external_eval_env = env
        if env is None:
            self._external_u_hist = None
            self._external_y_hist = None
            self._external_eval_needs_history_update = False
            return

        y = np.asarray(self._extract_phi_obs(obs), dtype=np.float64)
        self._external_u_hist = deque(
            [self._phi_u_eq.copy() for _ in range(self._phi_t_ini)],
            maxlen=self._phi_t_ini,
        )
        self._external_y_hist = deque(
            [y.copy() for _ in range(self._phi_t_ini + 1)],
            maxlen=self._phi_t_ini + 1,
        )
        self._external_eval_needs_history_update = False

    def _update_external_eval_history(self, obs):
        """Append the applied action and latest observation from the bound eval env."""
        if not self._external_eval_needs_history_update or self._external_eval_env is None:
            return

        u_applied = np.asarray(
            getattr(self._external_eval_env, "current_clipped_action", self._phi_u_eq),
            dtype=np.float64,
        ).reshape(-1)
        y = np.asarray(self._extract_phi_obs(obs), dtype=np.float64)
        self._external_u_hist.append(u_applied.copy())
        self._external_y_hist.append(y.copy())

    def _external_z_past(self):
        """Return the controller-side external evaluation history vector."""
        if self._external_u_hist is None or self._external_y_hist is None:
            raise RuntimeError("External Phi evaluation history is not initialised.")
        u_stack = np.stack(list(self._external_u_hist)).reshape(-1)
        y_stack = np.stack(list(self._external_y_hist)).reshape(-1)
        return np.concatenate([u_stack, y_stack])

    def reset_before_run(self, obs, info=None, env=None):
        """Reset Phi controller state for a fresh external evaluation episode."""
        super().reset_before_run(obs, info, env)
        self._bind_external_eval_env(obs, env)

    def select_action(self, obs, info=None):
        """Select an action, bridging raw-env evaluation into Phi history space."""
        if self._external_eval_env is None:
            return super().select_action(obs, info)

        self._update_external_eval_history(obs)
        actor_info = info
        if actor_info is None:
            actor_info = [{"current_step": 0, "x_ref": self._external_eval_env.X_GOAL}]
        elif isinstance(actor_info, dict):
            actor_info = [actor_info]

        with torch.no_grad():
            action = self.agent.ac.act(
                self._extract_phi_obs(obs),
                info=actor_info,
                z_past_raw=self._external_z_past(),
            )
        self._external_eval_needs_history_update = True
        return action

    def run(self, env=None, render=False, n_episodes=1, verbose=False):
        """Eval loop — passes z_past from PhiEnv to the actor at each step."""
        self.agent.reset()
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        self.z_past_normalizer.set_read_only()
        if env is None:
            env = self.eval_venv

        obs, _ = env.reset()
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths, ep_rmse = [], [], []
        frames = []
        agent_info = [{"current_step": 0, "x_ref": env.X_GOAL}]
        mse = []

        while len(ep_returns) < n_episodes:
            with torch.no_grad():
                action = self.agent.ac.act(
                    obs, info=agent_info, z_past_raw=env.z_past
                )
            obs, _, done, info = env.step(action)
            mse.append(info["mse"])
            if render:
                frames.append(env.render("rgb_array"))
            if verbose:
                print(f"obs {obs} | act {action}")
            if done:
                assert "episode" in info
                ep_rmse.append(np.array(mse).mean() ** 0.5)
                mse = []
                ep_returns.append(info["episode"]["r"])
                ep_lengths.append(info["episode"]["l"])
                obs, _ = env.reset()
                info["current_step"] = 0
                self.agent.reset()
            obs = self.obs_normalizer(obs)
            agent_info[0] = {"current_step": info["current_step"], "x_ref": env.X_GOAL}

        eval_results = {
            "ep_returns": np.asarray(ep_returns),
            "ep_lengths": np.asarray(ep_lengths),
            "rmse": np.array(ep_rmse).mean(),
            "rmse_std": np.array(ep_rmse).std(),
        }
        if frames:
            eval_results["frames"] = frames
        if len(env.queued_stats) > 0:
            eval_results.update({k: np.asarray(v) for k, v in env.queued_stats.items()})
        return eval_results

    def train_step(self):
        """Training step — overrides PPO_MPC.train_step() to wire z_past into the critic."""
        
        # Reset controlled-side MPC warm start and set PyTorch modules in training mode
        self.agent.reset()
        self.agent.train()
        
        # Allow normalization stats to be updated during rollout collection
        self.obs_normalizer.unset_read_only() # not currently used
        self.z_past_normalizer.unset_read_only()

        # Prepare PPO buffer
        rollouts = PPOBuffer(
            self.venv.observation_space, # (n_y,), no velocities!
            self.venv.action_space,
            self.rollout_steps,
            self.rollout_batch_size,
            z_past_dim=self._z_past_dim,
        )

        obs = self.obs
        start = time.time()

        # Collect the current z_past from each env and batch them
        z_past_raw = np.stack([env.z_past for env in self.venv.envs])
        
        # Actor uses raw z_past; critic uses normalized z_past
        z_past_norm = self.z_past_normalizer(z_past_raw)

        # Collect agent info across envs
        agent_info = []
        for env in self.venv.envs:
            agent_info.append(
                {"current_step": env.ctrl_step_counter, "x_ref": env.X_GOAL}
            )

        # Collect one time step from each env
        for _ in range(self.rollout_steps):
            # Freeze the current history batch for this timestep before stepping the envs
            z_past_raw_buf = z_past_raw
            z_past_norm_buf = z_past_norm
            
            # Query the actor-critic: sample action, estimate value, keep MPC solve info
            with torch.no_grad():
                act, v, logp, soln_info, results_dict, optimal = self.agent.ac.step(
                    torch.FloatTensor(obs).to(self.device),
                    info=agent_info,
                    z_past_norm=torch.FloatTensor(z_past_norm_buf).to(self.device),
                    z_past_raw=z_past_raw_buf,
                )

            # Step the environment, get observation and reward
            next_obs, rew, done, info = self.venv.step(act)
            next_obs = self.obs_normalizer(next_obs) # Currently unused (should probably remain so)
            rew = self.reward_normalizer(rew, done) # Currently unused (should probably use)

            # Fetch new input-output histories for the envs
            z_past_raw = np.stack([env.z_past for env in self.venv.envs])
            z_past_norm = self.z_past_normalizer(z_past_raw)

            # TODO: Survival bonus was a trick I tried, maybe no need for it anymore?
            # NB! IT IS IMPORTANT THAT I RETHINK IT IF I ADD REWARD NORMALIZATION
            survival_bonus = getattr(self, "survival_bonus", 0.0)
            if survival_bonus != 0.0:
                rew = rew + survival_bonus * (1 - done.astype(float))

            # Did the episode end? 0 if yes
            mask = 1 - done.astype(float)
            
            # Do not pretend future value was zero just because the episode was cut off by time limit
            terminal_v = np.zeros_like(v)

            for idx, inf in enumerate(info["n"]):
                # Update agent_info
                agent_info[idx] = {
                    "current_step": inf["current_step"],
                    "x_ref": self.venv.envs[idx].X_GOAL,
                }
                if done[idx]:
                    self.agent.reset(idx)
                    agent_info[idx]["current_step"] = 0

                if "terminal_info" not in inf:
                    continue
                inff = inf["terminal_info"]

                if "TimeLimit.truncated" in inff and inff["TimeLimit.truncated"]:
                    _t_z_past = self.venv.envs[idx]._terminal_z_past
                    # Normalize the terminal z_past using the current running
                    # statistics, but do not update those statistics here.
                    # This branch is only value-target bookkeeping for a
                    # truncated episode, not new rollout data for the critic
                    # normalizer to learn from.
                    _was_read_only = self.z_past_normalizer.read_only
                    self.z_past_normalizer.set_read_only()
                    try:
                        _t_z_past_norm = self.z_past_normalizer(_t_z_past)
                    finally:
                        if not _was_read_only:
                            self.z_past_normalizer.unset_read_only()
                    terminal_val = (
                        self.agent.ac.critic(
                            torch.FloatTensor(_t_z_past_norm[None]).to(self.device)
                        )
                        .squeeze().detach().cpu().numpy()
                    )
                    terminal_v[idx] = terminal_val

            rollouts.push({
                "obs": obs,
                "z_past": z_past_norm_buf,
                "act": act,
                "rew": rew,
                "mask": mask,
                "v": v,
                "logp": logp,
                "terminal_v": terminal_v,
                "info": soln_info,
                "results_dict": results_dict,
                "optimal": optimal,
            })
            obs = next_obs

        self.obs = obs
        self.total_steps += self.rollout_batch_size * self.rollout_steps

        # Critic value of the final rollout state
        last_val = (
            self.agent.ac.critic(torch.FloatTensor(z_past_norm).to(self.device))
            .detach().cpu().numpy()
        )

        # Compute returns and advantages
        ret, adv = compute_returns_and_advantages(
            rollouts.rew, rollouts.v, rollouts.mask, rollouts.terminal_v,
            last_val, gamma=self.gamma, use_gae=self.use_gae,
            gae_lambda=self.gae_lambda,
        )
        
        # Store return targets in the buffer
        rollouts.ret = ret
        
        # Store normalized advantages in the buffer
        rollouts.adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # Update results
        results = defaultdict(list)
        results["train"] = self.agent.update(rollouts, self.device)
        results["step"] = self.total_steps
        results["elapsed_time"] = time.time() - start
        return results

    def save(self, path):
        """Extends PPO_MPC.save() to also persist the z_past normalizer."""
        super().save(path)
        state = torch.load(path, weights_only=False)
        state["z_past_normalizer"] = self.z_past_normalizer.state_dict()
        torch.save(state, path)

    def load(self, path):
        """Extends PPO_MPC.load() to also restore the z_past normalizer."""
        super().load(path)
        state = torch.load(path, weights_only=False)
        if "z_past_normalizer" in state:
            self.z_past_normalizer.load_state_dict(state["z_past_normalizer"])

    def _expected_eval_ep_length(self) -> int:
        """Return the full-horizon eval episode length in control steps."""
        return int(getattr(self.eval_venv, "CTRL_STEPS"))

    def _is_full_length_eval(self, eval_results) -> bool:
        """Whether every eval episode reached the time limit without early termination."""
        ep_lengths = np.asarray(eval_results["ep_lengths"])
        if ep_lengths.size == 0:
            return False
        return np.all(np.isclose(ep_lengths, self._expected_eval_ep_length()))

    def learn(self, env=None, **kwargs):
        """Training loop with early stopping.

        Identical to PPO_MPC.learn() except: stops training when the eval
        score has not improved for ``early_stopping_patience`` consecutive
        evaluations.  Set early_stopping_patience to 0 (default) to disable.
        """
        
        patience = getattr(self, "early_stopping_patience", 0) # TODO: never used, remove? 
        evals_without_improvement = 0

        # Initial evaluation
        if self.eval_interval:
            results = defaultdict(list)
            eval_results = self.run(env=self.eval_venv, n_episodes=self.eval_batch_size)
            self.logger.info(
                "Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}".format(
                    eval_results["ep_lengths"].mean(),
                    eval_results["ep_lengths"].std(),
                    eval_results["ep_returns"].mean(),
                    eval_results["ep_returns"].std(),
                )
            )
            results.update({"step": self.total_steps, "elapsed_time": 0})
            results["eval"] = eval_results
            self.log_step(results)
            
        # Checkpoint logic TODO: remove, is not used 
        if self.num_checkpoints > 0:
            step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
            interval_save = np.zeros_like(step_interval, dtype=bool)

        # Main training loop: collect a rollout batch, do one PPO update over that batch, evaluate and log
        while self.total_steps < self.max_env_steps:
            # train_step() is overridden, see above 
            results = self.train_step()

            # Checkpoint.
            if self.total_steps >= self.max_env_steps or (
                self.save_interval and self.total_steps % self.save_interval == 0
            ):
                self.save(self.checkpoint_path)
                self.logger.info(f"Checkpoint | {self.checkpoint_path}")
                path = os.path.join(
                    self.output_dir,
                    "checkpoints",
                    "model_{}.pt".format(self.total_steps),
                )
                self.save(path)
            if self.num_checkpoints > 0:
                interval_id = np.argmin(
                    np.abs(np.array(step_interval) - self.total_steps)
                )
                if interval_save[interval_id] is False:
                    path = os.path.join(
                        self.output_dir, "checkpoints", f"model_{self.total_steps}.pt"
                    )
                    self.save(path)
                    interval_save[interval_id] = True

            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(
                    env=self.eval_venv, n_episodes=self.eval_batch_size
                )
                results["eval"] = eval_results
                self.logger.info(
                    "Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}".format(
                        eval_results["ep_lengths"].mean(),
                        eval_results["ep_lengths"].std(),
                        eval_results["ep_returns"].mean(),
                        eval_results["ep_returns"].std(),
                    )
                )
                # Save best model only when eval episodes run to completion.
                if self._is_full_length_eval(eval_results):
                    eval_score = eval_results["rmse"]
                    eval_best_score = getattr(self, "eval_best_score", float("inf"))
                    if self.eval_save_best and eval_score < eval_best_score:
                        self.eval_best_score = eval_score
                        self.save(os.path.join(self.output_dir, "model_best.pt"))
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1
                else:
                    evals_without_improvement = 0
                    self.logger.info(
                        "Skip best-model save: eval episode length %.2f < full length %d"
                        % (eval_results["ep_lengths"].mean(), self._expected_eval_ep_length())
                    )

                # Early stopping.
                if patience > 0 and evals_without_improvement >= patience:
                    self.logger.info(
                        f"Early stopping at step {self.total_steps}: no improvement "
                        f"for {patience} evals ({patience * self.eval_interval} steps)."
                    )
                    print(
                        f"\n*** Early stopping at step {self.total_steps} "
                        f"(no improvement for {patience} evals) ***\n",
                        flush=True,
                    )
                    break

            # Logging.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

        # Write experiment summary at end of training.
        self._write_experiment_summary()

    def _plot_trajectory(self, step: int, rmse: float):
        """Run one deterministic episode and save trajectory PNG + detailed metrics.

        Called from log_step() whenever an eval result is present.
        Computes rmse_xz, rmse_xz_theta, max_abs_theta, mean_action_util
        and logs them via self.logger.add_scalars(prefix="stat_eval").
        """
        # Save and switch to eval mode
        agent_was_training = self.agent.ac.training
        self.agent.eval()

        # Reset single-env I/O history so the episode starts clean
        mpc = self.agent.ac.actor.mpc
        mpc.u_hist = None
        mpc.y_hist = None

        # Run one episode, collecting full obs and actions
        obs, _ = self.eval_venv.reset()
        obs = self.obs_normalizer(obs)
        agent_info = [{"current_step": 0, "x_ref": self.eval_venv.X_GOAL}]

        traj_obs = [np.asarray(obs).copy()]
        traj_actions = []
        done = False
        with torch.no_grad():
            while not done:
                action = self.agent.ac.act(obs, info=agent_info, z_past_raw=self.eval_venv.z_past)
                traj_actions.append(np.asarray(action).flatten())
                obs, _, done, info = self.eval_venv.step(action)
                obs = self.obs_normalizer(obs)
                traj_obs.append(np.asarray(obs).copy())
                agent_info[0] = {
                    "current_step": info["current_step"],
                    "x_ref": self.eval_venv.X_GOAL,
                }

        traj_obs = np.array(traj_obs)          # (T+1, obs_dim)
        traj_actions = np.array(traj_actions)   # (T, act_dim)
        x_goal = np.asarray(self.eval_venv.X_GOAL)  # (T, nx)

        # Compute detailed RMSE metrics
        T = min(len(traj_obs), len(x_goal))
        # traj_obs cols: [x, z, theta] (PhiEnv filters y_indices=[0,2,4])
        # x_goal   cols: [x, xdot, z, zdot, theta, thetadot] (full 6D reference)
        err_x = traj_obs[:T, 0] - x_goal[:T, 0]
        err_z = traj_obs[:T, 1] - x_goal[:T, 2]
        err_theta = traj_obs[:T, 2] - x_goal[:T, 4]

        rmse_xz = float(np.sqrt(np.mean(err_x**2 + err_z**2)))
        rmse_xz_theta = float(np.sqrt(np.mean(err_x**2 + err_z**2 + err_theta**2)))
        max_abs_theta = float(np.max(np.abs(traj_obs[:, 2])))

        # Action utilisation: fraction of [lo, hi] range used (0-1 scale)
        act_lo = np.asarray(self.eval_venv.action_space.low)
        act_hi = np.asarray(self.eval_venv.action_space.high)
        act_range = np.where((act_hi - act_lo) > 0, act_hi - act_lo, 1.0)
        mean_action_util = float(((traj_actions - act_lo) / act_range).mean())

        # Log metrics
        self.logger.add_scalars(
            {
                "rmse_xz": rmse_xz,
                "rmse_xz_theta": rmse_xz_theta,
                "max_abs_theta": max_abs_theta,
                "mean_action_util": mean_action_util,
            },
            step,
            prefix="stat_eval",
        )
        print(f"  Traj  | rmse_xz {rmse_xz:.4f}  rmse_xzt {rmse_xz_theta:.4f}"
              f"  max_theta {max_abs_theta:.3f}  act_util {mean_action_util:.3f}",
              flush=True)

        # Plot
        traj_x, traj_z = traj_obs[:, 0], traj_obs[:, 1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(traj_x, traj_z, "r--", label="Controller")
        ax.plot(x_goal[:, 0], x_goal[:, 2], "g--", label="Reference")
        ax.scatter(traj_x[0], traj_z[0], color="g", marker="o", s=100,
                   label="Initial State")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Step {step:,}  |  RMSE_xz {rmse_xz:.4f}  RMSE_xzt {rmse_xz_theta:.4f}")
        ax.legend(loc="upper right")
        ax.set_box_aspect(0.5)
        fig.tight_layout()

        save_path = os.path.join(self.output_dir, f"traj_{step:07d}.png")
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"  Traj  | saved → {save_path}", flush=True)

        # Restore training mode
        if agent_was_training:
            self.agent.train()

    def _write_experiment_summary(self):
        """Write experiment_summary.json with start vs end parameters."""
        cost_p = self.agent.ac.actor.cost_param.detach().cpu().numpy()
        phi_p = self.agent.ac.actor.phi_param.detach().cpu().numpy()
        off = self.n_y + self.n_u

        def _cost_dict(cp):
            return {
                "Qy": cp[:self.n_y].tolist(),
                "R": cp[self.n_y:off].tolist(),
                "tw": float(cp[off]),
                "ws": float(cp[off + 1]),
                "tm": float(cp[off + 2]),
            }

        summary = {
            "total_steps": self.total_steps,
            "cost_param_initial": _cost_dict(self._initial_cost_param),
            "cost_param_final": _cost_dict(cost_p),
            "phi_norm_initial": self._initial_phi_norm,
            "phi_norm_final": float(np.linalg.norm(phi_p)),
            "best_eval_rmse": getattr(self, "_best_eval_rmse", None),
            "best_eval_rmse_full_length": getattr(
                self, "_best_eval_rmse_full_length", None
            ),
        }

        path = os.path.join(self.output_dir, "experiment_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Experiment summary saved -> {path}")

    def log_step(self, results):
        """Log training progress.

        Identical to PPO_MPC.log_step() except the final print:
        prints phi norm + cost params instead of the full ~17k-entry mpc_param.
        """
        step = results["step"]
        self.logger.add_scalars(
            {
                "step": step,
                "step_time": results["elapsed_time"],
                "progress": step / self.max_env_steps,
            },
            step,
            prefix="time",
        )
        if "train" in results:
            self.logger.add_scalars(
                {
                    k: results["train"][k]
                    for k in [
                        "policy_loss", "value_loss", "entropy_loss",
                        "approx_kl", "theta_loss", "ref_loss", "valid_frac",
                    ]
                },
                step,
                prefix="loss",
            )
            ep_lengths = np.asarray(self.venv.length_queue)
            ep_returns = np.asarray(self.venv.return_queue)
            ep_cv = np.asarray(self.venv.queued_stats["constraint_violation"])
            self.logger.add_scalars(
                {
                    "ep_length": ep_lengths.mean(),
                    "ep_return": ep_returns.mean(),
                    "ep_return_std": ep_returns.std(),
                    "ep_reward": (ep_returns / ep_lengths).mean(),
                    "ep_constraint_violation": ep_cv.mean(),
                },
                step,
                prefix="stat",
            )
            total_cv = self.venv.accumulated_stats["constraint_violation"]
            self.logger.add_scalars(
                {"constraint_violation": total_cv}, step, prefix="stat"
            )
        if "eval" in results:
            er = results["eval"]
            self.logger.add_scalars(
                {
                    "ep_length": er["ep_lengths"].mean(),
                    "ep_return": er["ep_returns"].mean(),
                    "ep_return_std": er["ep_returns"].std(),
                    "ep_reward": (er["ep_returns"] / er["ep_lengths"]).mean(),
                    "constraint_violation": er["constraint_violation"].mean(),
                    "rmse": er["rmse"],
                    "rmse_std": er["rmse_std"],
                },
                step,
                prefix="stat_eval",
            )
            # Track best RMSE for experiment summary.
            if er["rmse"] < getattr(self, "_best_eval_rmse", float("inf")):
                self._best_eval_rmse = er["rmse"]
            if self._is_full_length_eval(er) and er["rmse"] < getattr(
                self, "_best_eval_rmse_full_length", float("inf")
            ):
                self._best_eval_rmse_full_length = er["rmse"]

        # Trajectory plot + detailed metrics (before dump)
        if "eval" in results:
            self._plot_trajectory(step, results["eval"]["rmse"])

        # Terminal progress report
        pct = 100.0 * step / self.max_env_steps
        print(f"\n{'='*60}", flush=True)
        print(f"  Step {step:>7,} / {self.max_env_steps:,}  ({pct:.1f}%)", flush=True)
        print(f"{'='*60}", flush=True)
        if "eval" in results:
            er = results["eval"]
            print(f"  Eval  | RMSE {er['rmse']:.4f} (+/-{er['rmse_std']:.4f})"
                  f"  return {er['ep_returns'].mean():.1f}"
                  f"  length {er['ep_lengths'].mean():.0f}"
                  f"  cv {er['constraint_violation'].mean():.0f}", flush=True)
        if "train" in results:
            tr = results["train"]
            print(f"  Train | policy_loss {tr['policy_loss']:.4f}"
                  f"  value_loss {tr['value_loss']:.4f}"
                  f"  kl {tr['approx_kl']:.4f}", flush=True)
            if "kkt_norm_mean" in tr:
                print(f"  KKT   | mean {tr['kkt_norm_mean']:.3e}"
                      f"  min {tr['kkt_norm_min']:.3e}"
                      f"  max {tr['kkt_norm_max']:.3e}", flush=True)
                self.logger.add_scalars(
                    {"kkt_norm_mean": tr["kkt_norm_mean"],
                     "kkt_norm_max": tr["kkt_norm_max"]},
                    step, prefix="loss",
                )
        cost_p = self.agent.ac.actor.cost_param.detach().numpy()
        phi_p = self.agent.ac.actor.phi_param.detach().numpy()
        logstd_val = float(self.agent.ac.actor.logstd.detach().mean().numpy())
        off = self.n_y + self.n_u
        print(f"  Theta | Qy {cost_p[:self.n_y]}  R {cost_p[self.n_y:off]}"
              f"  tw {cost_p[off]:.3f}  ws {cost_p[off+1]:.2f}  tm {cost_p[off+2]:.3f}"
              f"  std {np.exp(logstd_val):.4f}  phi_norm {np.linalg.norm(phi_p):.4f}",
              flush=True)

        # Log individual cost params, phi norm, and exploration std.
        self.logger.add_scalars(
            {
                "Qy_x": float(cost_p[0]),
                "Qy_z": float(cost_p[1]),
                "Qy_theta": float(cost_p[2]),
                "R_u1": float(cost_p[self.n_y]),
                "R_u2": float(cost_p[self.n_y + 1]),
                "tw": float(cost_p[off]),
                "ws": float(cost_p[off + 1]),
                "tm": float(cost_p[off + 2]),
                "phi_norm": float(np.linalg.norm(phi_p)),
                "logstd": logstd_val,
                "std": float(np.exp(logstd_val)),
            },
            step,
            prefix="cost_param",
        )
        self.logger.dump_scalars()
        print(f"{'='*60}\n", flush=True)
