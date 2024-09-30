from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
import casadi as cs
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from multiprocessing import Pool

from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.math_and_models.neural_networks import MLP


class TD3MPC:
    """An MPC-based policy function approximator with TD3 critic"""

    def __init__(self,
                 env_fun,
                 obs_space,
                 act_space,
                 gamma,
                 model,
                 horizon: int = 5,
                 q_mpc: list = [2],
                 r_mpc: list = [1],
                 warmstart: bool = True,
                 soft_constraints: bool = True,
                 constraint_tol: float = 1e-6,
                 additional_constraints: list = None,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 hidden_dim=256,
                 eps=0.1,
                 tau=0.005,
                 activation='relu',
                 device=None,
                 ):
        """Creates task and controller.

        Args:
            model (Callable): dynamics function for the task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            constraint_tol (float): Tolerance to add the constraint as sometimes solvers are not exact.
            additional_constraints (list): List of additional constraints
        """
        self.env = env_fun
        # params
        self.obs_space = obs_space
        self.act_space = act_space
        low, high = act_space.low, act_space.high
        self.action_space_low = torch.FloatTensor(low).to(device)
        self.action_space_high = torch.FloatTensor(high).to(device)
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.activation = activation
        self.device = device

        # Actor setup
        self.actor = MPCPolicyFunction(env_fun, gamma, model, horizon, q_mpc, r_mpc, warmstart,
                                       soft_constraints,
                                       constraint_tol,
                                       additional_constraints,
                                       )

        # Critic setup
        self.critic = MLPQFunction(obs_space, act_space, hidden_dims=[hidden_dim] * 2, activation=self.activation)
        self.critic_targ = MLPQFunction(obs_space, act_space, hidden_dims=[hidden_dim] * 2, activation=self.activation)

        # optimizers
        self.update_step_count = 0
        self.update_freq = 3
        self.tau = tau
        self.actor_lr = actor_lr
        # self.actor_opt = torch.optim.Adam([self.actor.theta_param_], actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic.q_net1.parameters())
                                           + list(self.critic.q_net2.parameters()), critic_lr)

        # Initial reset for function approximator
        self.reset()

    def reset(self):
        self.actor.reset()

    def select_action(self, obs, info=None, mode='eval'):
        action, info, results_dict = self.actor.select_action(obs, info=info, mode=mode)
        if mode == 'train':
            action += self.eps * (np.random.random(self.act_space.shape[0]) - 0.5)
        return action, info, results_dict

    def compute_q_loss(self, batch):
        """Returns q-value loss(es) given batch of data."""
        obs, act, rew, next_obs, mask, info = (batch['obs'], batch['act'], batch['rew'], batch['next_obs'],
                                               batch['mask'], batch['info'])
        q1, q2 = self.critic.forward(obs, act)

        with torch.no_grad():
            # Next action
            next_act, _ = self.actor.select_action_batch(next_obs, info)
            noise = (0.5*torch.randn_like(next_act)).clamp(-0.2, 0.2)
            next_act = (next_act+noise).clamp(self.action_space_low, self.action_space_high)
            # Target Q value for next state and next action
            next_q1_targ, next_q2_targ = self.critic_targ.forward(next_obs, next_act)
            next_q_targ = torch.min(next_q1_targ, next_q2_targ)
            # q value regression target
            q_targ = rew + self.gamma * mask * next_q_targ

        q1_loss = (q1 - q_targ).pow(2).mean()
        q2_loss = (q2 - q_targ).pow(2).mean()
        critic_loss = q1_loss + q2_loss
        return critic_loss

    def update(self, batch, batch_th):
        """Updates model parameters based on current training batch."""
        results = defaultdict(list)

        # critic update
        critic_loss = self.compute_q_loss(batch_th)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.update_step_count % self.update_freq == 0:
            # actor update
            action_batch, nabla_pi_batch = self.actor.select_action_batch(batch['obs'], batch['info'],
                                                                          sensitivity_compute=True)
            action_batch.requires_grad_()
            q1, _ = self.critic.forward(batch_th['obs'], action_batch)
            policy_loss = q1.mean()
            policy_loss.backward()
            grad_q = action_batch.grad.unsqueeze(1).numpy()
            delj = np.matmul(grad_q, nabla_pi_batch).squeeze(1)
            self.actor.solver_dict['theta_param_val'] += self.actor_lr * delj.mean(axis=0)
            print(self.actor.solver_dict['theta_param_val'])
            results['policy_loss'] = policy_loss.item()

            # update target networks
            soft_update(self.critic, self.critic_targ, self.tau)

        results['critic_loss'] = critic_loss.item()
        self.update_step_count += 1
        return results


class MLPQFunction(nn.Module):

    def __init__(self, obs_space, act_space, hidden_dims, activation):
        super().__init__()
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
            discrete = False
        else:
            raise NotImplementedError

        self.q_net1 = MLP(obs_dim + act_dim, 1, hidden_dims, activation)
        self.q_net2 = MLP(obs_dim + act_dim, 1, hidden_dims, activation)

    def forward(self, obs, act):
        return self.q_net1(torch.cat([obs, act], dim=-1)), self.q_net2(torch.cat([obs, act], dim=-1))

class MPCPolicyFunction:
    def __init__(self,
                 env_fun,
                 gamma,
                 model,
                 horizon: int = 5,
                 q_mpc: list = [2],
                 r_mpc: list = [1],
                 warmstart: bool = True,
                 soft_constraints: bool = True,
                 constraint_tol: float = 1e-6,
                 additional_constraints: list = None,
                 ):
        self.env = env_fun
        self.model = model
        self.dt = self.model.dt
        self.T = horizon
        self.gamma = gamma
        self.q_mpc, self.r_mpc = q_mpc, r_mpc
        self.target_q_mpc, self.target_r_mpc = q_mpc, r_mpc
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        self.update_step_count = 0
        self.soft_constraints = soft_constraints
        self.constraint_tol = constraint_tol
        self.warmstart = warmstart

        # Multi-processing
        self.multi_pool = Pool(8)

        # Constraint list
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            (self.constraints, self.state_constraints_sym,
             self.input_constraints_sym) = reset_constraints(self.env.constraints.constraints
                                                             + self.additional_constraints)
        else:
            (self.constraints, self.state_constraints_sym,
             self.input_constraints_sym) = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []

        # Additional entries
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.mode = None
        self.traj = None
        self.traj_step = 0

        # Setup optimizer
        self.solver_dict, self.qfn_dict = None, None
        self.dynamics_func = None
        self.set_dynamics_func()
        self.setup_optimizer()
        self._init_theta_val()
        self.temp = 0

    def reset(self):
        # Previously solved states & inputs, useful for warm start.
        self.u_prev = None
        self.x_prev = None
        self.sigma_prev = None
        self.x_goal = None
        self.traj = None
        self.traj_step = 0
        # self.X_EQ = self.env.X_EQ
        # self.U_EQ = self.env.U_EQ

        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0

    def add_constraints(self, constraints):
        """Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        (self.constraints, self.state_constraints_sym,
         self.input_constraints_sym) = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self, constraints):
        """Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        """
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(
            old_constraints_list)

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency."""
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)

    def compute_initial_guess(self, init_state, goal_states, x_lin, u_lin):
        """Use LQR to get an initial guess of the """
        dfdxdfdu = self.model.df_func(x=x_lin, u=u_lin)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        lqr_gain, _, _ = compute_discrete_lqr_gain_from_cont_linear_system(dfdx, dfdu, self.Q, self.R, self.dt)

        x_guess = np.zeros((self.model.nx, self.T + 1))
        u_guess = np.zeros((self.model.nu, self.T))
        x_guess[:, 0] = init_state

        for i in range(self.T):
            u = lqr_gain @ (x_guess[:, i] - goal_states[:, i]) + u_lin
            u_guess[:, i] = u
            x_guess[:, i + 1, None] = self.dynamics_func(x0=x_guess[:, i], p=u)['xf'].toarray()

        return x_guess, u_guess

    def setup_optimizer(self):
        """Sets up nonlinear optimization problem."""
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        etau = 1e-5

        # Define optimizer and variables.
        # States.
        x_var = cs.MX.sym("x_var", nx, T + 1)
        # Inputs.
        u_var = cs.MX.sym("u_var", nu, T)
        # Add slack variables
        state_slack = cs.MX.sym("sigma_var", nx, T + 1)
        opt_vars = cs.vertcat(cs.reshape(u_var, -1, 1),
                              cs.reshape(x_var, -1, 1),
                              cs.reshape(state_slack, -1, 1))
        opt_vars_fn = cs.Function("opt_vars_fun", [opt_vars], [x_var, u_var, state_slack])

        # Parameters
        # Fixed parameters
        # Initial state.
        x_init = cs.MX.sym("x_init", nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = cs.MX.sym("x_ref", nx, T + 1)
        fixed_param = cs.vertcat(x_init, cs.reshape(x_ref, -1, 1))

        # Learnable parameters
        Q, th_q, nq = _create_semi_definite_matrix(nx)
        R, th_r, nr = _create_semi_definite_matrix(nu)
        # theta_param = cs.MX.sym("theta_var", nq + nr)
        theta_param = cs.vertcat(th_q, th_r)

        # cost (cumulative)
        cost = 0
        w = 1e3 * np.ones((1, nx))
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += self.gamma ** i * cost_func(x=x_var[:, i],
                                                u=u_var[:, i],
                                                Xr=x_ref[:, i],
                                                Ur=np.zeros((nu, 1)),
                                                Q=Q,
                                                R=R)['l']
        # Terminal cost.
        cost += self.gamma ** T * cost_func(x=x_var[:, -1],
                                            u=np.zeros((nu, 1)),
                                            Xr=x_ref[:, -1],
                                            Ur=np.zeros((nu, 1)),
                                            Q=Q,
                                            R=R)['l']
        # Constraints
        g, hu, hx, hs = [], [], [], []
        # initial condition constraints
        g.append(x_var[:, 0] - x_init)
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            g.append(x_var[:, i + 1] - next_state)
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                cost += w @ state_slack[:, i]
                hx.append(state_constraint(x_var[:, i])[:nx] - state_slack[:, i])
                hx.append(state_constraint(x_var[:, i])[nx:] - state_slack[:, i])
                hs.append(-state_slack[:, i])
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                hu.append(input_constraint(u_var[:, i]) + self.constraint_tol)
        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            cost += w @ state_slack[:, -1]
            hx.append(state_constraint(x_var[:, -1])[:nx] - state_slack[:, -1])
            hx.append(state_constraint(x_var[:, -1])[nx:] - state_slack[:, -1])
            hs.append(-state_slack[:, -1])
        # Setting casadi constraints and bounds
        G = cs.vertcat(*g)
        Hu = cs.vertcat(*hu)
        Hx = cs.vertcat(*hx)
        Hs = cs.vertcat(*hs)
        constraint_exp = cs.vertcat(*g, *hu, *hx, *hs)
        lbg = [0] * G.shape[0] + [-np.inf] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])
        ubg = [0] * G.shape[0] + [0] * (Hu.shape[0] + Hx.shape[0] + Hs.shape[0])
        lbg = cs.vertcat(*lbg)
        ubg = cs.vertcat(*ubg)

        # Create solver (IPOPT solver in this version)
        opts_setting = {
            "ipopt.max_iter": 50,
            "ipopt.print_level": 5,
            "print_time": 5,
            "ipopt.mu_target": etau,
            "ipopt.mu_init": etau,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.acceptable_obj_change_tol": 1e-4,
        }
        vnlp_prob = {
            "f": cost,
            "x": opt_vars,
            "p": cs.vertcat(fixed_param, theta_param),
            "g": constraint_exp,
        }
        vsolver = cs.nlpsol("vsolver", "ipopt", vnlp_prob, opts_setting)

        # Sensitivity
        # Multipliers
        lamb = cs.MX.sym("lambda", G.shape[0])
        mu_u = cs.MX.sym("muu", Hu.shape[0])
        mu_x = cs.MX.sym("mux", Hx.shape[0])
        mu_s = cs.MX.sym("mus", Hs.shape[0])
        mult = cs.vertcat(lamb, mu_u, mu_x, mu_s)

        # Build Lagrangian
        lagrangian = (
                cost
                + cs.transpose(lamb) @ G
                + cs.transpose(mu_u) @ Hu
                + cs.transpose(mu_x) @ Hx
                + cs.transpose(mu_s) @ Hs
        )
        lagrangian_fn = cs.Function("Lag", [opt_vars, mult, fixed_param, theta_param], [lagrangian])

        # Generate sensitivity of the Lagrangian
        dlag_fn = lagrangian_fn.factory(
            "dlag_fn",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i3"],
        )
        dlag_dw, dlag_dtheta = dlag_fn(opt_vars, mult, fixed_param, theta_param)
        dlag_dw_fn = cs.Function('dlag_dw_fn', [opt_vars, mult, fixed_param, theta_param], [dlag_dw])
        dlag_dtheta_fn = cs.Function('dlag_dtheta_fn', [opt_vars, mult, fixed_param, theta_param], [dlag_dtheta])

        # Build KKT matrix
        R_kkt = cs.vertcat(
            cs.transpose(dlag_dw),
            G,
            mu_u * Hu + etau,
            mu_x * Hx + etau,
            mu_s * Hs + etau,
        )

        # z contains all variables of the lagrangian
        z = cs.vertcat(opt_vars, lamb, mu_u, mu_x, mu_s)

        # Generate sensitivity of the KKT matrix
        Rfun = cs.Function("Rfun", [z, fixed_param, theta_param], [R_kkt])
        dR_sensfunc = Rfun.factory("dR", ["i0", "i1", "i2"], ["jac:o0:i0", "jac:o0:i2"])
        [dRdz, dRdP] = dR_sensfunc(z, fixed_param, theta_param)

        # Generate sensitivity of the optimal solution
        dzdP = -cs.inv(dRdz) @ dRdP
        dPi = cs.Function("dPi", [z, fixed_param, theta_param], [dzdP[: nu, :]])

        self.solver_dict = {
            'solver': vsolver,
            'x_var': x_var,
            'u_var': u_var,
            'state_slack': state_slack,
            'opt_vars': opt_vars,
            'opt_vars_fn': opt_vars_fn,
            'x_init': x_init,
            'x_ref': x_ref,
            'theta_param': theta_param,
            'cost': cost,
            'lower_bound': lbg,
            'upper_bound': ubg,
            'dlag_dw_fn': dlag_dw_fn,
            'dlag_dtheta_fn': dlag_dtheta_fn,
            'dpi_fn': dPi
        }

    def get_references(self, traj_step=None):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            if traj_step is None:
                traj_step = self.traj_step
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(traj_step, self.traj.shape[-1])
            end = min(traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def _init_theta_val(self):
        # self.theta_param_ = torch.FloatTensor(np.concatenate((self.q_mpc, self.r_mpc)))
        # self.target_theta_param_ = torch.FloatTensor(np.concatenate((self.q_mpc, self.r_mpc)))
        # self.theta_param_.requires_grad_()
        self.solver_dict['theta_param_val'] = np.concatenate((self.q_mpc, self.r_mpc)).copy()
        # self.solver_dict['target_theta_param_val'] = np.concatenate((self.q_mpc, self.r_mpc)).copy()

    def select_action(self, obs, info=None, mode='eval'):
        """Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info
            mode (string): Current mode of evaluation (eval vs train)

        Returns:
            action (ndarray): Input/action to the task/env.
        """

        solver_dict = self.solver_dict
        solver = solver_dict['solver']

        # Collect the fixed param
        # Assign reference trajectory within horizon.
        goal_states = self.get_references(self.traj_step)
        fixed_param = np.concatenate((obs[:self.model.nx, None], goal_states.T.reshape(-1, 1)))
        # Collect learnable parameters
        theta_param = solver_dict['theta_param_val'][:, None]
        if self.mode == 'tracking':
            self.traj_step += 1

        opt_vars_init = np.zeros((solver_dict['opt_vars'].shape[0], solver_dict['opt_vars'].shape[1]))
        # if self.warmstart and self.x_prev is None and self.u_prev is None:
        #    x_guess, u_guess = self.compute_initial_guess(obs, goal_states, self.X_EQ, self.U_EQ)
        #    opti.set_initial(x_var, x_guess)
        #    opti.set_initial(u_var, u_guess) # Initial guess for optimization problem.
        # elif self.warmstart and self.x_prev is not None and self.u_prev is not None:
        if self.warmstart and self.x_prev is not None and self.u_prev is not None:
            # shift previous solutions by 1 step
            opt_vars_init = update_initial_guess(self.x_prev, self.u_prev, self.sigma_prev)

        # Solve the optimization problem.
        soln = solver(
            x0=opt_vars_init,
            p=np.concatenate((fixed_param, theta_param))[:, 0],
            lbg=solver_dict['lower_bound'],
            ubg=solver_dict['upper_bound'],
        )

        # Post-processing the solution
        opt_vars = soln['x'].full()
        x_val, u_val, sigma_val = solver_dict['opt_vars_fn'](opt_vars)
        self.x_prev = x_val.full()
        self.u_prev = u_val.full()
        self.sigma_prev = sigma_val.full()
        results_dict = {'horizon_states': deepcopy(self.x_prev), 'horizon_inputs': deepcopy(self.u_prev),
                        'goal_states': deepcopy(goal_states)}
        # results_dict['t_wall'].append(opti.stats()['t_wall_total'])

        # Take the first action from the solved action sequence.
        if self.u_prev.ndim > 1:
            action = self.u_prev[:, 0]
        else:
            action = np.array([self.u_prev[0]])

        # additional info
        info = {
            'success': solver.stats()['success'],
            'soln': deepcopy(soln),
            'fixed_param': deepcopy(fixed_param),
            'theta_param': deepcopy(theta_param),
            'traj_step': deepcopy(self.traj_step) - 1
        }
        return action, info, results_dict

    def select_action_batch(self, obs_batch, info_batch, update_guess=False, sensitivity_compute=False):
        solver_dict = self.solver_dict
        solver = solver_dict['solver']
        lbg = solver_dict['lower_bound']
        ubg = solver_dict['upper_bound']
        theta_param = solver_dict['theta_param_val'][:, None]
        dpi = solver_dict['dpi_fn']
        opt_vars_fn = solver_dict['opt_vars_fn']

        eval_data_batch = []
        for obs, info in zip(obs_batch, info_batch):
            traj_step, soln = info['traj_step'], info['soln']
            goal_states = self.get_references(traj_step)
            fixed_param = np.concatenate((obs[:self.model.nx, None], goal_states.T.reshape(-1, 1)))

            # shift previous solutions by 1 step
            opt_vars_init = soln['x'].full()
            if update_guess:
                x_prev, u_prev, sigma_prev = solver_dict['opt_vars_fn'](opt_vars_init)
                x_prev, u_prev, sigma_prev = x_prev.full(), u_prev.full(), sigma_prev.full()
                opt_vars_init = update_initial_guess(x_prev, u_prev, sigma_prev)

            temp = [solver, opt_vars_init, fixed_param, theta_param, lbg, ubg, opt_vars_fn, dpi, sensitivity_compute]
            eval_data_batch.append(temp)
        soln_batch = self.multi_pool.map(_select_action, eval_data_batch)
        p()

        action_batch = []
        nabla_pi_batch = []
        for soln in soln_batch:
            action, nabla_pi = soln
            action_batch.append(action)
            nabla_pi_batch.append(nabla_pi)
        action_batch = torch.FloatTensor(np.array(action_batch))
        return action_batch, nabla_pi_batch


class ReplayBuffer(object):
    '''Storage for replay buffer during training.

    Attributes:
        max_size (int): maximum size of the replay buffer.
        batch_size (int): number of samples (steps) per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self, obs_space, act_space, max_size, batch_size=None):
        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size

        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.pos = 0
        self.buffer_size = 0
        N = max_size
        self.scheme = {
            'obs': {
                'vshape': (N, *obs_dim)
            },
            'next_obs': {
                'vshape': (N, *obs_dim)
            },
            'act': {
                'vshape': (N, act_dim)
            },
            'rew': {
                'vshape': (N, 1)
            },
            'mask': {
                'vshape': (N, 1),
                'init': np.ones
            },
            'info': {
                'vshape': N
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocate space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            if k == 'info':
                self.__dict__[k] = deque([], maxlen=info['vshape'])
            else:
                vshape = info['vshape']
                dtype = info.get('dtype', np.float32)
                init = info.get('init', np.zeros)
                self.__dict__[k] = init(vshape, dtype=dtype)
        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        '''Returns current size of the buffer.'''
        return self.buffer_size

    def state_dict(self):
        '''Returns a snapshot of current buffer.'''
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self, state):
        '''Restores buffer from previous state.'''
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.'''
        # batch size
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            if k == 'info':
                self.__dict__[k].append(v)
            else:
                shape = self.scheme[k]['vshape'][1:]
                dtype = self.scheme[k].get('dtype', np.float32)
                v_ = np.asarray(v, dtype=dtype).reshape((n,) + shape)

                if self.pos + n <= self.max_size:
                    self.__dict__[k][self.pos:self.pos + n] = v_
                else:
                    # wrap around
                    remain_n = self.pos + n - self.max_size
                    self.__dict__[k][self.pos:self.max_size] = v_[:-remain_n]
                    self.__dict__[k][:remain_n] = v_[-remain_n:]

        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)
        self.pos = (self.pos + n) % self.max_size

    def sample(self, batch_size=None, device=None):
        '''Returns data batch.'''
        if batch_size is None:
            batch_size = self.batch_size
        batch = {}
        batch_th = {}
        indices = np.random.randint(0, len(self), size=batch_size)
        for k, info in self.scheme.items():
            if k == 'info':
                temp = []
                for i in indices:
                    temp.append(self.__dict__[k][i])
                batch[k] = temp
                batch_th[k] = temp
            else:
                shape = info['vshape'][1:]
                v = self.__dict__[k].reshape(-1, *shape)[indices]
                batch[k] = v
                if device is None:
                    batch_th[k] = torch.as_tensor(v)
                else:
                    batch_th[k] = torch.as_tensor(v, device=device)
        return batch, batch_th


# -----------------------------------------------------------------------------------
#                   Misc
# -----------------------------------------------------------------------------------


def update_initial_guess(x_prev, u_prev, sigma_prev):
    # shift previous solutions by 1 step
    u_guess = deepcopy(u_prev)
    x_guess = deepcopy(x_prev)
    sigma_guess = deepcopy(sigma_prev)
    u_guess[:, :-1] = u_guess[:, 1:]
    x_guess[:, :-1] = x_guess[:, 1:]
    sigma_guess[:, :-1] = sigma_guess[:, 1:]
    opt_vars_init = np.concatenate((u_guess.T.reshape(-1, 1),
                                    x_guess.T.reshape(-1, 1),
                                    sigma_guess.T.reshape(-1, 1)))
    return opt_vars_init


def _select_action(eval_data):
    solver, opt_vars_init, fixed_param, theta_param, lbg, ubg, opt_vars_fn, dpi, sensitivity_compute = eval_data

    # Solve the optimization problem.
    soln = solver(
        x0=opt_vars_init,
        p=np.concatenate((fixed_param, theta_param))[:, 0],
        lbg=lbg,
        ubg=ubg,
    )
    opt_vars = soln['x'].full()
    _, u_val, _ = opt_vars_fn(opt_vars)
    u_val = u_val.full()
    if u_val.ndim > 1:
        action = u_val[:, 0]
    else:
        action = np.array([u_val[0]])

    # Sensitivity computation
    if sensitivity_compute:
        mult = soln['lam_g'].full()
        z = np.concatenate((opt_vars, mult), axis=0)
        nabla_pi = dpi(z, fixed_param, theta_param).full()
    else:
        nabla_pi = np.zeros((u_val.shape[0], theta_param.shape[0]))
    return action, nabla_pi


def _create_semi_definite_matrix(n):
    # U = cs.SX.sym("U", cs.Sparsity.lower(n))
    # u = cs.vertcat(*U.nonzeros())
    # W_upper = cs.Function("Lower_tri_W", [u], [U])
    # np = int(n * (n + 1) / 2)
    # p = cs.MX.sym("p", np)
    # W = W_upper(p)
    # WW = W.T @ W

    np = n
    P = cs.MX.sym("P", n)
    W = cs.diag(P)
    WW = cs.sqrt(W.T @ W)
    return WW, P, np

def soft_update(source, target, tau):
    """Synchronizes target networks with exponential moving average."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(source, target):
    """Synchronizes target networks by copying over parameters directly."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
