'''Model Predictive Control using Sequential Quadratic Programming (SQP).'''
import time
from copy import deepcopy

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

        self.X_EQ = self.env.X_GOAL
        self.U_EQ = self.env.U_GOAL
        self.init_step_solver = 'ipopt' # for nonlinear warmstart
        self.qp_solver = 'qrqp'


    
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

        # TODO: when setup optimizer, first call compute_initial_guess
        # then setup_optimizer
        # self.setup_optimizer(solver=self.init_step_solver)

        # Previously solved states & inputs, useful for warm start.
        # nominal solution
        self.x_prev = None 
        self.u_prev = None
        # previous delta solution
        self.dx_prev = None
        self.du_prev = None

        # self.setup_sqp_optimizer(solver=self.qp_solver)
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
        self.setup_optimizer(solver=self.init_step_solver)
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

        self.x_prev, self.u_prev = x_val, u_val
        time_after = time.time()
        print('MPC _compute_initial_guess time: ', time_after - time_before)

    
    def compute_linearized_dynamics(self, x, u):
        '''Compute the linearized dynamics around the current state and input.'''
        dfdxdfdu = self.model.df_func(x=x, u=u)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        Ad, Bd = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        return Ad, Bd

    def setup_sqp_optimizer(self, init_state, solver='qrqp'):
        '''Setup the optimizer for the SQP MPC.'''
        # create the refernce trajectory for linearization
        # print('env.__dir__()', self.env.__dir__())
        
        if self.x_prev is None or self.u_prev is None:
            # TODO: compute the initial guess in select_action
            self.compute_initial_guess(init_state, self.get_references())
        x_guess, u_guess = self.x_prev, self.u_prev
        
        nx, nu = self.model.nx, self.model.nu
        T = self.T

        # Define the optimizer and variables.
        if self.qp_solver in ['qrqp', 'qpoases']:
            opti = cs.Opti('conic')
        else:
            opti = cs.Opti()

        # states
        dx_var = opti.variable(nx, T+1)
        # inputs
        du_var = opti.variable(nu, T)
        # initial state
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables 
        state_slack = opti.variable(len(self.state_constraints_sym))
        input_slack = opti.variable(len(self.input_constraints_sym))

        # cost 
        # quadratic term
        S, state_cost_lin, input_cost_lin = get_cost(self.R, self.Q, T)
        # linear term
        f_1 = 2 * input_cost_lin @ u_guess # input cost
        f_2 = 2 * input_cost_lin @ (x_guess - x_ref) # state cost
        H = 2 * S
        f = cs.vertcat(f_1, f_2)
        assert f_1.shape[0] == nu * T
        assert f_2.shape[0] == nx * (T + 1) 
        assert H.shape[0] == nx * (T + 1) + nu * T 
        assert H.shape[0] == H.shape[1]

        dz = cs.vertcat(du_var, dx_var)
        cost = 0.5 * cs.mtimes(dz.T, cs.mtimes(H, dz)) + cs.mtimes(f.T, dz)


        # # linearized system dynamics constraints
        # A_bar = np.zeros((nx * (T + 1), nx * (T + 1)))
        # B_bar = np.zeros((nx * (T + 1), nu * T))
        # for i in range(T):
        #     Ad, Bd = self.compute_linearized_dynamics(x_guess[:, i], u_guess[:, i])
        #     A_bar[i * nx:(i + 1) * nx, (i) * nx:(i + 1) * nx] = Ad
        #     B_bar[i * nx:(i + 1) * nx, i * nu:(i + 1) * nu] = Bd

        # A_bar = np.block([[np.zeros(nx, nx*T), np.zeros((nx))],
        #                   [A_bar,        np.zeros((nx * T, nx))]])
        # B_bar = np.block([np.zeros((nx, nu * T)), B_bar])
        # assert A_bar.shape[0] == nx * (T + 1)
        # assert B_bar.shape[0] == nx * (T + 1) and B_bar.shape[1] == nu * T
        # H_bar = np.zeros((nx * (T + 1), nx))
        # H_bar[:nx, :] = np.eye(nx)
        # delta_x_init = x_init - x_guess # enforce the initial state constraint
        # Aeq = np.concatenate([-B_bar, np.eye(nx * (T + 1))-A_bar], axis=1)
        # beq = H_bar @ delta_x_init
        # opti.subject_to(cs.mtimes(Aeq, cs.vertcat(du_var, dx_var)) == beq)

        for i in range(self.T):
            # dynamics constraints
            Ad, Bd = self.compute_linearized_dynamics(x_guess[:, i], u_guess[:, i])
            next_dx = Ad @ dx_var[:, i] + Bd @ du_var[:, i]
            opti.subject_to(dx_var[:, i + 1] == next_dx)

            # state and input constraints
            soft_con_coeff = 10
            for sc_i, state_constraints in enumerate(self.state_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(state_constraints(dx_var[:, i] + x_guess[:, i]) <= state_slack[sc_i])
                    cost += soft_con_coeff * state_slack[sc_i]**2
                    opti.subject_to(state_slack[sc_i] >= 0)
                else:
                    opti.subject_to(state_constraints(dx_var[:, i] + x_guess[:, i]) <= -self.constraint_tol)
            
            for ic_i, input_constraints in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraints(du_var[:, i] + u_guess[:, i]) <= input_slack[ic_i])
                    cost += soft_con_coeff * input_slack[ic_i]**2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(input_constraints(du_var[:, i] + u_guess[:, i]) <= -self.constraint_tol)

        # final state constraint
        for sc_i, state_constraints in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraints(dx_var[:, -1] + x_guess[:, -1]) <= state_slack[sc_i])
                cost += soft_con_coeff * state_slack[sc_i]**2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraints(dx_var[:, -1] + x_guess[:, -1]) <= -self.constraint_tol)

        # initial condition constraints
        opti.subject_to(dx_var[:, 0] + x_guess[:, 0] == x_init)
        opti.minimize(cost)
        opts = {'expand': True}
        opti.solver(solver, opts)
        self.opti_dict = {'opti': opti,
                          'dx_var': dx_var,
                          'du_var': du_var,
                          'x_init': x_init,
                          'x_ref': x_ref,
                          'cost': cost
                          }
        
    def select_action(self, 
                        obs, 
                        info=None
                        ):
        '''Solve qp to get the optimal action.

        Args:
            obs (np.array): current observation.
            info (dict): additional information.
        Returns:
            action (np.array): Input/action to the task/env.
        '''
        self.setup_sqp_optimizer(obs, solver=self.qp_solver)

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        dx_var = opti_dict['dx_var']
        du_var = opti_dict['du_var']
        x_init = opti_dict['x_init']
        x_ref = opti_dict['x_ref']

        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.get_references()
        opti.set_value(x_ref, goal_states)
        if self.env.TASK == Task.TRAJ_TRACKING:
            self.traj_step += 1
        if self.warmstart and self.dx_prev is not None and self.du_prev is not None:
            opti.set_initial(dx_var, self.dx_prev)
            opti.set_initial(du_var, self.du_prev)

        # try:
        sol = opti.solve()
        dx_val, du_val = sol.value(dx_var), sol.value(du_var)
        self.dx_prev, self.du_prev = dx_val, du_val
        # store the actual solution directly in x_prev and u_prev
        self.u_prev = self.u_prev + du_val
        self.x_prev = self.x_prev + dx_val
        self.results_dict['horizon_states'].append(self.x_prev)
        self.results_dict['horizon_inputs'].append(self.u_prev)
        # except RuntimeError as e:
        #     print(e)
        #     return_status = opti.return_status()
        #     if return_status == 'unknown':
        #         self.terminate_loop = True
        #         du_val = self.du_prev
        #         if du_val is None:
        #             print('[WARN]: MPC Infeasible first step.')
        #             du_val = np.zeros((self.model.nu, self.T))
        #     elif return_status == 'Maximum_Iterations_Exceeded':
        #         self.terminate_loop = True
        #         du_val = opti.debug.value(du_var)
        #     elif return_status == 'Search_Direction_Becomes_Too_Small':
        #         self.terminate_loop = True
        #         du_val = opti.debug.value(du_var)
            
        # take first one from solved action sequence
        if du_val.ndim > 1:
            action = du_val[:, 0]
        else:
            action = np.array([du_val[0]])
        action += self.u_prev[:, 0]
        self.prev_action = action
        return action
        
        
        