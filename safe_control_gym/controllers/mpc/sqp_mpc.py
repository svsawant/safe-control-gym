'''Model Predictive Control using Sequential Quadratic Programming (SQP).'''
import time
from copy import deepcopy
from sys import platform

import casadi as cs
import numpy as np

from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list
from safe_control_gym.controllers.mpc.sqp_mpc_utils import get_cost

class SQPMPC(MPC):

    '''Model Predictive Control using Sequential Quadratic Programming (SQP).'''

    def __init__(
            self,
            env_func,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            warmstart: bool = True,
            soft_constraints: bool = False,
            terminate_run_on_done: bool = True,
            constraint_tol: float = 1e-6,
            # runner args
            # shared/base args
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
        '''
        # Store all params/args
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})
        super().__init__(
            env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=soft_constraints,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            # prior_info=prior_info,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            **kwargs
        )

        # Task
        self.env = env_func()
        if additional_constraints is not None:
            additional_constraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_constraintsList.constraints
            self.constraints, self.state_constraints_sym, self.input_constraints_sym \
                                                        = reset_constraints(self.env.constraints.constraints
                                                                            + self.additional_constraints)
        else:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym \
                                                        = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []
        # Model parameters
        self.model = self.get_prior(self.env)
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)

        # boolean flags

        self.soft_constraints = soft_constraints
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done

        # self.X_EQ = self.env.X_GOAL
        # self.U_EQ = self.env.U_GOAL
        self.init_step_solver = 'ipopt' # for nonlinear warmstart
        self.qp_solver = 'qrqp'

    def set_lin_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        # Original version, used in shooting.
        delta_x = cs.MX.sym('delta_x', self.model.nx, 1)
        delta_u = cs.MX.sym('delta_u', self.model.nu, 1)
        x_guess = cs.MX.sym('x_guess', self.model.nx, 1)
        u_guess = cs.MX.sym('u_guess', self.model.nu, 1)
        dfdxdfdu = self.model.df_func(x=x_guess, u=u_guess)
        dfdx = dfdxdfdu['dfdx']#.toarray()
        dfdu = dfdxdfdu['dfdu']#.toarray()
        # x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        # self.linear_dynamics_func = cs.integrator(
        #    'linear_discrete_dynamics', self.model.integration_algo,
        #    {
        #        'x': delta_x,
        #        'p': delta_u,
        #        'ode': x_dot_lin_vec
        #    }, {'tf': self.dt}
        # )
        # Ad, Bd = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        Ad = cs.DM_eye(self.model.nx) + dfdx * self.dt
        Bd = dfdu * self.dt
        x_dot_lin = Ad @ delta_x + Bd @ delta_u
        self.linear_dynamics_func = cs.Function('linear_discrete_dynamics',
                                                [delta_x, delta_u, x_guess, u_guess],
                                                [x_dot_lin],
                                                ['x0', 'p', 'x_guess', 'u_guess'],
                                                ['xf'])
        self.dfdx = dfdx
        self.dfdu = dfdu
    
    def reset(self):
        '''Prepares for training or evaluation.'''

        print('==========Resetting the controller.==========')
        # Setup reference input .
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0
        # Dynamics model
        self.set_dynamics_func()
        self.set_lin_dynamics_func()

        # Previously solved states & inputs, useful for warm start.
        # nominal solution
        self.x_prev = None 
        self.u_prev = None
        # # previous delta solution
        # self.dx_prev = None
        # self.du_prev = None
        self.x_guess = None
        self.u_guess = None

        init_state = self.env.reset()
        # print('init_state', init_state)
        self.setup_optimizer()
        self.setup_sqp_optimizer()
        self.setup_results_dict()
    

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)
    
    def compute_initial_guess(self, init_state, goal_states):
        time_before = time.time()
        '''Use IPOPT to get an initial guess of the '''
        self.setup_optimizer()
        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var'] # optimization variables
        u_var = opti_dict['u_var'] # optimization variables
        x_init = opti_dict['x_init'] # initial state
        x_ref = opti_dict['x_ref'] # reference state/trajectory
        # Assign the initial state.
        opti.set_value(x_init, init_state) # initial state should have dim (nx,)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.mode == 'tracking':
            self.traj_step += 1
         # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
        except RuntimeError:
            print('=============Warm-starting fails=============')
            x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)

        self.x_guess, self.u_guess = x_val, u_val
        self.x_prev, self.u_prev = x_val, u_val
        time_after = time.time()
        print('MPC _compute_initial_guess time: ', time_after - time_before)
        self.setup_sqp_optimizer()

    def setup_sqp_optimizer(self):
        '''Sets up convex optimization problem.

        Including cost objective, variable bounds and dynamics constraints.
        '''
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        if self.qp_solver in ['qrqp', 'qpoases']:
            opti = cs.Opti('conic')
        else:
            opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)

        x_guess_var = opti.parameter(nx, T + 1)
        u_guess_var = opti.parameter(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            cost += cost_func(x=x_var[:, i] + x_guess_var[:, i],
                              u=u_var[:, i] + u_guess_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1] + x_guess_var[:, -1],
                          u=np.zeros((nu, 1)) + u_guess_var[:, -1],
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=self.Q,
                          R=self.R)['l']
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.linear_dynamics_func(x0=x_var[:, i], p=u_var[:, i], 
                                                   x_guess=x_guess_var[:,i], u_guess=u_guess_var[:,i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)
            # State and input constraints
            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                opti.subject_to(state_constraint(x_var[:, i] + x_guess_var[:, i]) <= -self.constraint_tol)

            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                opti.subject_to(input_constraint(u_var[:, i] + u_guess_var[:, i]) <= -self.constraint_tol)

        # final state constraints
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            opti.subject_to(state_constraint(x_var[:, -1] + x_guess_var[:, -1]) <= -self.constraint_tol)

        # initial condition constraints
        opti.subject_to(x_var[:, 0] + x_guess_var[:, 0] == x_init)
        opti.minimize(cost)
        # create solver (IPOPT solver for now )
        opts = {'expand': True}
        # if platform == 'linux':
        #     opts.update({'print_time': 1, 'print_header': 0})
        #     opti.solver(self.solver, opts)
        # elif platform == 'darwin':
        #     opts.update({'ipopt.max_iter': 100})
        #     opti.solver('ipopt', opts)
        # else:
        #     print('[ERROR]: CasADi solver tested on Linux and OSX only.')
        #     exit()
        opti.solver(self.qp_solver, opts)
        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_guess_var': x_guess_var,
            'u_guess_var': u_guess_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost
        }
        
    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solve nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info.

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        if self.x_guess is None or self.u_guess is None:
            self.compute_initial_guess(obs, self.get_references())

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']
        u_var = opti_dict['u_var']
        x_guess_var = opti_dict['x_guess_var']
        u_guess_var = opti_dict['u_guess_var']
        x_init = opti_dict['x_init']
        x_ref = opti_dict['x_ref']

        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        opti.set_value(x_guess_var, self.x_guess)
        opti.set_value(u_guess_var, self.u_guess)
        if self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
        if self.warmstart and self.u_prev is not None and self.x_prev is not None:
            opti.set_initial(x_var, self.x_prev)
            opti.set_initial(u_var, self.u_prev)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
            self.x_prev = x_val
            self.u_prev = u_val
            self.results_dict['horizon_states'].append(deepcopy(self.x_prev) + self.x_guess)
            self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev) + self.u_guess)
        except RuntimeError as e:
            print(e)
            return_status = opti.return_status()
            if return_status == 'unknown':
                self.terminate_loop = True
                u_val = self.u_prev
                if u_val is None:
                    print('[WARN]: MPC Infeasible first step.')
                    u_val = np.zeros((1, self.model.nu))
            elif return_status == 'Maximum_Iterations_Exceeded':
                self.terminate_loop = True
                u_val = opti.debug.value(u_var)
            elif return_status == 'Search_Direction_Becomes_Too_Small':
                self.terminate_loop = True
                u_val = opti.debug.value(u_var)
            # x_val = self.x_prev

        # take first one from solved action sequence
        if u_val.ndim > 1:
            action = u_val[:, 0]
        else:
            action = np.array([u_val[0]])
        action += self.u_guess[0]
        self.prev_action = action
        self.x_guess = x_val + self.x_guess
        self.u_guess = u_val + self.u_guess
        return action
        
        
        