"""Q learning for Model Predictive Control."""

from copy import deepcopy

import casadi as cs
import numpy as np
import os
import time
from collections import defaultdict

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.rlmpc.q_mpc_utils import QMPC, ReplayBuffer
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import is_wrapped


class Q_MPC(BaseController):
    """MPC with full nonlinear model."""

    def __init__(
            self,
            env_func,
            # runner args
            # shared/base args
            mpc_config,
            training=True,
            checkpoint_path='model_latest.pt',
            output_dir: str = 'results/temp',
            use_gpu: bool = False,
            seed: int = 0,
            terminate_run_on_done: bool = True,
            **kwargs
    ):
        """Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            output_dir (str): output directory to write logs and results.
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
        """
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})
        self.initial_eval = True

        # Task.
        self.env = env_func(seed=seed)
        self.env = RecordEpisodeStatistics(self.env, self.deque_size)
        self.eval_env = env_func(seed=seed * 111)
        self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
        self.terminate_run_on_done = terminate_run_on_done

        # Agent
        model = self.get_prior(self.env)
        self.agent = QMPC(self.env, model, **mpc_config)

        # logging
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # disable logging to texts and tfboard for testing
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

    def close(self):
        """Cleans up resources."""
        self.env.close()

    def reset(self):
        """Prepares for training or evaluation."""
        # MPC agent reset
        self.agent.reset()

        # Result dict for collecting results
        self.setup_results_dict()
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, _ = self.env.reset()
            self.obs = obs
            self.buffer = ReplayBuffer(self.env.observation_space, self.env.action_space,
                                       self.max_buffer_size, self.train_batch_size)
        else:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def select_action(self, obs, info=None, soln_info=None, mode='eval'):
        """Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        """
        # Solve the optimization problem.
        action, additional_info, results_dict = self.agent.select_action(obs, mode=mode)
        self.results_dict.update(results_dict)
        self.prev_action = action
        if soln_info is not None:
            soln_info.update(additional_info)
        return action

    def setup_results_dict(self):
        """Setup results dictionary to store run information."""
        self.results_dict = {'obs': [],
                             'reward': [],
                             'done': [],
                             'info': [],
                             'action': [],
                             'horizon_inputs': [],
                             'horizon_states': [],
                             'goal_states': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': []
                             }

    def reset_before_run(self, obs, info=None, env=None):
        """Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        """
        self.reset()

    def learn(self, env=None, **kwargs):
        """Performs learning (pre-training, training, fine-tuning, etc.)."""
        if self.num_checkpoints > 0:
            step_interval = np.linspace(0, self.max_env_steps, self.num_checkpoints)
            interval_save = np.zeros_like(step_interval, dtype=bool)
        while self.total_steps < self.max_env_steps:
            results = self.train_step()

            # checkpoint
            if self.total_steps >= self.max_env_steps or (
                    self.save_interval and self.total_steps % self.save_interval == 0):
                # latest/final checkpoint
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
                path = os.path.join(self.output_dir, 'checkpoints', 'model_{}.pt'.format(self.total_steps))
                self.save(path)
            if self.num_checkpoints > 0:
                interval_id = np.argmin(np.abs(np.array(step_interval) - self.total_steps))
                if interval_save[interval_id] is False:
                    # Intermediate checkpoint.
                    path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                    self.save(path, save_buffer=False)
                    interval_save[interval_id] = True

            # eval
            if self.eval_interval and (self.initial_eval or self.total_steps % self.eval_interval == 0):
                self.initial_eval = False
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | '
                                 'ep_return {:.3f} +/- {:.3f}'.format(eval_results['ep_lengths'].mean(),
                                                                      eval_results['ep_lengths'].std(),
                                                                      eval_results['ep_returns'].mean(),
                                                                      eval_results['ep_returns'].std()))
                # save best model
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))

            # logging
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def train_step(self, **kwargs):
        """Performs a training step."""
        # self.agent.train()
        obs = self.obs
        start = time.time()

        # for _ in range(self.rollout_steps):
        soln_info = {}
        action = self.select_action(obs, soln_info=soln_info, mode='train')
        next_obs, rew, done, info = self.env.step(action)
        mask = 1 - np.asarray(done)

        # time truncation is not true termination
        terminal_idx, terminal_obs = [], []
        # for idx, inf in enumerate(info['n']):
        if 'terminal_info' in info:
            inff = info['terminal_info']
            if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                terminal_obs.append(info['terminal_observation'])
            elif inff['out_of_bounds']:
                terminal_obs.append(info['terminal_observation'])
        # if len(terminal_obs) > 0:
        #     terminal_obs = _unflatten_obs(self.obs_normalizer(_flatten_obs(terminal_obs)))

        # collect the true next states and masks (accounting for time truncation and out of bounds)
        true_next_obs = next_obs.copy()
        true_mask = mask.copy()
        # for idx, term_ob in zip(terminal_idx, terminal_obs):
        #     true_next_obs[idx] = term_ob
        #     true_mask[idx] = 1.0
        # true_next_obs = _flatten_obs(true_next_obs)

        experience = {
            'obs': obs[None, :],
            'act': action[None, :],
            'rew': np.array([rew]),
            'next_obs': true_next_obs[None, :],
            'mask': np.array([true_mask]),
            'info': soln_info
        }
        self.buffer.push(experience)
        obs = next_obs
        if done:
            self.agent.reset()
            obs, _ = self.env.reset()
        self.obs = obs
        self.total_steps += 1

        # learn
        results = defaultdict(list)
        if self.total_steps > self.warm_up_steps and not self.total_steps % self.train_interval:
            # Regardless of how long you wait between updates,
            # the ratio of env steps to gradient steps is locked to 1.
            # alternatively, can update once each step
            for j in range(self.train_interval):
                batch = self.buffer.sample(self.train_batch_size, self.device)
                res = self.agent.update(batch)
                for k, v in res.items():
                    results[k].append(v)
            print(self.agent.param_dict)
        results = {k: sum(v) / len(v) for k, v in results.items()}
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def run(self,
            env=None,
            render=False,
            n_episodes=1,
            verbose=False,
            logging=False,
            max_steps=None,
            terminate_run_on_done=None
            ):
        """Runs evaluation with current policy.

        Args:
            render (bool): if to do real-time rendering.
            logging (bool): if to log on terminal.

        Returns:
            dict: evaluation statistics, rendered frames.
        """
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, info = env.reset()
        self.agent.reset()
        ep_returns, ep_lengths, eval_return = [], [], 0.0
        frames = []
        mse, ep_rmse_mean, ep_rmse_std = [], [], []
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)
            obs, _, done, info = env.step(action)
            mse.append(info["mse"])
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            if verbose:
                print(f'obs {obs} | act {action}')
            if done:
                assert 'episode' in info
                ep_rmse_mean.append((np.array(mse) ** 0.5).mean())
                ep_rmse_std.append((np.array(mse) ** 0.5).std())
                mse = []
                ep_returns.append(info['episode']['r'])
                ep_lengths.append(info['episode']['l'])
                obs, _ = env.reset()
                self.agent.reset()
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths,
                        'rmse': np.array(ep_rmse_mean).mean(),
                        'rmse_std': np.array(ep_rmse_std).mean()}
        if len(frames) > 0:
            eval_results['frames'] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def log_step(self, results):
        """Does logging after a training step."""
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'time': results['elapsed_time'],
                'progress': step / self.max_env_steps,
            },
            step,
            prefix='time')

        # learning stats
        if 'td_error' in results:
            self.logger.add_scalars(
                {
                    k: results[k]
                    for k in ['td_error']
                },
                step,
                prefix='loss')

        # performance stats
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        ep_constraint_violation = np.asarray(self.env.queued_stats['constraint_violation'])
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_return_std': ep_returns.std(),
                'ep_reward': (ep_returns / ep_lengths).mean(),
                'ep_constraint_violation': ep_constraint_violation.mean()
            },
            step,
            prefix='stat')

        # total constraint violation during learning
        total_violations = self.env.accumulated_stats['constraint_violation']
        self.logger.add_scalars({'constraint_violation': total_violations}, step, prefix='stat')

        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            eval_constraint_violation = results['eval']['constraint_violation']
            eval_rmse = results['eval']['rmse']
            eval_rmse_std = results['eval']['rmse_std']
            self.logger.add_scalars(
                {
                    'ep_length': eval_ep_lengths.mean(),
                    'ep_return': eval_ep_returns.mean(),
                    'ep_return_std': eval_ep_returns.std(),
                    'ep_reward': (eval_ep_returns / eval_ep_lengths).mean(),
                    'constraint_violation': eval_constraint_violation.mean(),
                    'rmse': eval_rmse,
                    'rmse_std': eval_rmse_std
                },
                step,
                prefix='stat_eval')
        # print summary table
        self.logger.dump_scalars()
