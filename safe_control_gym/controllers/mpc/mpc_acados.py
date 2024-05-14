'''Model Predictive Control.'''
import time
from copy import deepcopy

import casadi as cs
import numpy as np
import scipy

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.mpc.mpc_utils import (compute_discrete_lqr_gain_from_cont_linear_system,
                                                        compute_state_rmse, get_cost_weight_matrix,
                                                        reset_constraints, rk_discrete)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS, create_constraint_list

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel

class MPC_ACADOS(BaseController):
    '''MPC with full nonlinear model.'''

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
            use_RTI: bool = False,
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
        super().__init__(env_func, output_dir, use_gpu, seed, **kwargs)
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        # if prior_info is None:
        #    self.prior_info = {}
        # Task.
        self.env = env_func()
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints + self.additional_constraints)
        else:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []
        # Model parameters
        self.model = self.get_prior(self.env)
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        # self.prior_info = prior_info

        self.soft_constraints = soft_constraints
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done

        # print(self.env.__dir__())
        # print('self.env.X_GOAL', self.env.X_GOAL)
        # NOTE: The naming X_EQ and U_EQ can be confusing 
        self.X_EQ = self.env.X_GOAL
        self.U_EQ = self.env.U_GOAL
        
        # warm-starting
        self.init_solver = 'ipopt'
        self.x_guess = None
        self.u_guess = None

        self.use_RTI = use_RTI
        self.set_dynamics_func()
        self.set_up_acados_ode_model()
        self.setup_acados_optimizer()
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp) # , \
                                    # json_file=f'acados_{self.ocp.model.name}.json')


    def add_constraints(self,
                        constraints
                        ):
        '''Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        '''
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self,
                           constraints
                           ):
        '''Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        '''
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(old_constraints_list)

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def reset(self):
        print('=================Resetting MPC=================')
        '''Prepares for training or evaluation.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            # Step along the reference.
            self.traj_step = 0
        # Dynamics model.
        self.set_dynamics_func()
        self.set_up_acados_ode_model
        # CasADi optimizer.
        # self.setup_optimizer()
        self.setup_acados_optimizer()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None
        self.setup_results_dict()

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        # self.dynamics_func = cs.integrator('fd', 'rk',
        #                                   {
        #                                    'x': self.model.x_sym,
        #                                    'p': self.model.u_sym,
        #                                    'ode': self.model.x_dot
        #                                    },
        #                                   {'tf': self.dt})
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)

    def set_up_acados_ode_model(self) -> AcadosModel:

        model_name = self.env.NAME
        
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        # acados_model.xdot = self.model.x_dot_acados # must be symbolic
        acados_model.u = self.model.u_sym
        acados_model.name = model_name

        # set up rk4 (acados need symbolic expression, not function
        k1 = self.model.fc_func(acados_model.x, acados_model.u)
        k2 = self.model.fc_func(acados_model.x + self.dt/2 * k1, acados_model.u)
        k3 = self.model.fc_func(acados_model.x + self.dt/2 * k2, acados_model.u)
        k4 = self.model.fc_func(acados_model.x + self.dt * k3, acados_model.u)
        f_disc = acados_model.x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        acados_model.disc_dyn_expr = f_disc
        # f_expl = self.model.x_dot
        # f_impl = self.model.x_dot_acados - f_expl
        # model.f_impl_expr = f_impl
        # model.f_expl_expr = f_expl

        # store meta information # NOTE: unit is missing
        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_ode_model = acados_model
        

    # def compute_lqr_initial_guess(self, init_state, goal_states, x_lin, u_lin):
    #     '''Use LQR to get an initial guess of the '''
    #     dfdxdfdu = self.model.df_func(x=x_lin, u=u_lin)
    #     dfdx = dfdxdfdu['dfdx'].toarray()
    #     dfdu = dfdxdfdu['dfdu'].toarray()
    #     lqr_gain, _, _ = compute_discrete_lqr_gain_from_cont_linear_system(dfdx, dfdu, self.Q, self.R, self.dt)
        
    #     # initialize the guess solutions
    #     x_guess = np.zeros((self.model.nx, self.T + 1))
    #     u_guess = np.zeros((self.model.nu, self.T))
    #     x_guess[:, 0] = init_state
    #     # add the lqr gain and states to the guess
    #     for i in range(self.T):
    #         u = lqr_gain @ (x_guess[:, i] - goal_states[:, i]) + u_lin
    #         u_guess[:, i] = u
    #         x_guess[:, i + 1, None] = self.dynamics_func(x0=x_guess[:, i], p=u)['xf'].toarray()
    #     return x_guess, u_guess
    
    def compute_initial_guess(self, init_state, goal_states):
        time_before = time.time()
        '''Use IPOPT to get an initial guess of the '''
        self.setup_optimizer(solver=self.init_solver)
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

        self.x_guess = x_val
        self.u_guess = u_val
        time_after = time.time()
        print('MPC _compute_initial_guess time: ', time_after - time_before)

    def setup_acados_optimizer(self):
        '''Sets up nonlinear optimization problem.'''
        print('=================Setting up acados optimizer=================')
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_ode_model

        # set dimensions
        ocp.dims.N = self.T # prediction horizon

        # set cost 
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx+nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # bounded input constraints
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = np.arange(nu)
        # bounded state constraints
        ocp.constraints.Jbx = np.eye(nx)
        ocp.constraints.lbx = self.env.constraints.state_constraints[0].lower_bounds
        ocp.constraints.ubx = self.env.constraints.state_constraints[0].upper_bounds
        ocp.constraints.idxbx = np.arange(nx)
        # bounded terminal state constraints
        ocp.constraints.Jbx_e = np.eye(nx)
        ocp.constraints.lbx_e = self.env.constraints.state_constraints[0].lower_bounds
        ocp.constraints.ubx_e = self.env.constraints.state_constraints[0].upper_bounds
        ocp.constraints.idxbx_e = np.arange(nx)

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 5000
        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        self.ocp = ocp

        # # Constraints
        # for i in range(self.T):
        #     # Dynamics constraints.
        #     next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
        #     opti.subject_to(x_var[:, i + 1] == next_state)

        #     for sc_i, state_constraint in enumerate(self.state_constraints_sym):
        #         if self.soft_constraints:
        #             opti.subject_to(state_constraint(x_var[:, i]) <= state_slack[sc_i])
        #             cost += 10000 * state_slack[sc_i]**2
        #             opti.subject_to(state_slack[sc_i] >= 0)
        #         else:
        #             opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
        #     for ic_i, input_constraint in enumerate(self.input_constraints_sym):
        #         if self.soft_constraints:
        #             opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
        #             cost += 10000 * input_slack[ic_i]**2
        #             opti.subject_to(input_slack[ic_i] >= 0)
        #         else:
        #             opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

        # # Final state constraints.
        # for sc_i, state_constraint in enumerate(self.state_constraints_sym):
        #     if self.soft_constraints:
        #         opti.subject_to(state_constraint(x_var[:, -1]) <= state_slack[sc_i])
        #         cost += 10000 * state_slack[sc_i] ** 2
        #         opti.subject_to(state_slack[sc_i] >= 0)
        #     else:
        #         opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        # # initial condition constraints
        # opti.subject_to(x_var[:, 0] == x_init)

        # opti.minimize(cost)
        # # Create solver (IPOPT solver in this version)
        # # opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        # # opts = {'expand': True}
        # opts = {'expand': True, 'error_on_fail': False}
        # # solver_opts = {'error_on_fail': False}
        # # opti.solver('ipopt', opts)
        # # opti.solver('sqpmethod', opts)
        # opti.solver(solver, opts)
        # # opti.solver('feasibleSQP', opts)

        # # print(opti)
        # # exit()
        
        # self.opti_dict = {
        #     'opti': opti,
        #     'x_var': x_var,
        #     'u_var': u_var,
        #     'x_init': x_init,
        #     'x_ref': x_ref,
        #     'cost': cost
        # }

    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        time_before = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx
        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, "lbx", obs)
        self.acados_ocp_solver.set(0, "ubx", obs)

        # warm-starting solver (otherwise, zeros by default)
        if self.warmstart:
            if self.x_guess is None or self.u_guess is None:   
                # compute initial guess with IPOPT 
                self.compute_initial_guess(obs, self.get_references())
            for idx in range(self.T + 1):
                init_x = self.x_guess[:, idx]
                self.acados_ocp_solver.set(idx, "x", init_x)
            for idx in range(self.T):
                if nu == 1:
                    init_u = np.array([self.u_guess[idx]])
                else:
                    init_u = self.u_guess[:, idx]
                self.acados_ocp_solver.set(idx, "u", init_u)

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1
        for idx in range(self.T):
            y_ref = np.concatenate((goal_states[:, idx], np.zeros((nu,))))
            self.acados_ocp_solver.set(idx, "yref", y_ref)
        y_ref_e = goal_states[:, -1] 
        self.acados_ocp_solver.set(self.T, "yref", y_ref_e)

        # solve the optimization problem
        if self.use_RTI:
            # preparation phase
            self.acados_ocp_solver.options_set('rti_phase', 1)
            status = self.acados_ocp_solver.solve()

            # feedback phase
            self.acados_ocp_solver.options_set('rti_phase', 2) 
            status = self.acados_ocp_solver.solve()
            action = self.acados_ocp_solver.get(0, "u")

        elif not self.use_RTI:
            status = self.acados_ocp_solver.solve()
            if status not in [0, 2]:
                self.acados_ocp_solver.print_statistics()
                raise Exception(f'acados returned status {status}. Exiting.')
                # print(f"acados returned status {status}. ")
            if status == 2:
                print(f"acados returned status {status}. ")
            action = self.acados_ocp_solver.get(0, "u")

        # opti_dict = self.opti_dict
        # opti = opti_dict['opti']
        # x_var = opti_dict['x_var'] # optimization variables
        # u_var = opti_dict['u_var'] # optimization variables
        # x_init = opti_dict['x_init'] # initial state
        # x_ref = opti_dict['x_ref'] # reference state/trajectory

        # # Assign the initial state.
        # opti.set_value(x_init, obs)
        # # Assign reference trajectory within horizon.
        # goal_states = self.get_references()
        # opti.set_value(x_ref, goal_states)
        # if self.mode == 'tracking':
        #     self.traj_step += 1

        # if self.warmstart and self.x_prev is None and self.u_prev is None:
        # #    x_guess, u_guess = self.compute_lqr_initial_guess(obs, goal_states, self.X_EQ, self.U_EQ)
        #    print(f'computing initial guess with {self.init_solver}')
        #    x_guess, u_guess = self.compute_initial_guess(obs, goal_states)
        #    opti.set_initial(x_var, x_guess)
        #    opti.set_initial(u_var, u_guess) # Initial guess for optimization problem.
        # elif self.warmstart and self.x_prev is not None and self.u_prev is not None:
        # # if self.warmstart and self.x_prev is not None and self.u_prev is not None:
        #     # shift previous solutions by 1 step
        #     x_guess = deepcopy(self.x_prev)
        #     u_guess = deepcopy(self.u_prev)
        #     x_guess[:, :-1] = x_guess[:, 1:]
        #     u_guess[:-1] = u_guess[1:]
        #     opti.set_initial(x_var, x_guess)
        #     opti.set_initial(u_var, u_guess)
        # # Solve the optimization problem.
        # try:
        #     sol = opti.solve()
        #     x_val, u_val = sol.value(x_var), sol.value(u_var)
        # except RuntimeError:
        #     print('=============Infeasible MPC Problem=============')
        #     x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
        # skip = 8
        # print('x_val: ', x_val[:,::skip])
        # print('u_val: ', u_val[::skip])
        # self.x_prev = x_val
        # self.u_prev = u_val
        # self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        # self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        # self.results_dict['goal_states'].append(deepcopy(goal_states))
        # self.results_dict['t_wall'].append(opti.stats()['t_wall_total'])
        # # Take the first action from the solved action sequence.
        # if u_val.ndim > 1:
        #     action = u_val[:, 0]
        # else:
        #     action = np.array([u_val[0]])
        # self.prev_action = action
        time_after = time.time()
        print('Acados MPC _select_action time: ', time_after - time_before)
        return action

    def get_references(self):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            goal_states = np.concatenate([
                self.traj[:, start:end],
                np.tile(self.traj[:, -1:], (1, remain))
            ], -1)
        else:
            raise Exception('Reference for this mode is not implemented.')
        return goal_states  # (nx, T+1).

    def setup_optimizer(self, solver='ipopt'):
        '''Sets up nonlinear optimization problem.'''
        print('=================Setting up MPC optimizer=================')
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables
        state_slack = opti.variable(len(self.state_constraints_sym))
        input_slack = opti.variable(len(self.input_constraints_sym))

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(x=x_var[:, i],
                              u=u_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1],
                          u=np.zeros((nu, 1)),
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=self.Q,
                          R=self.R)['l']
        # Constraints
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(state_constraint(x_var[:, i]) <= state_slack[sc_i])
                    cost += 10000 * state_slack[sc_i]**2
                    opti.subject_to(state_slack[sc_i] >= 0)
                else:
                    opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
                    cost += 10000 * input_slack[ic_i]**2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1]) <= state_slack[sc_i])
                cost += 10000 * state_slack[sc_i] ** 2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)
        opts = {'expand': True, 'error_on_fail': False}
        opti.solver(solver, opts)

        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost
        }

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
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

    def run(self,
            env=None,
            render=False,
            logging=False,
            max_steps=None,
            terminate_run_on_done=None
            ):
        '''Runs evaluation with current policy.

        Args:
            render (bool): if to do real-time rendering.
            logging (bool): if to log on terminal.

        Returns:
            dict: evaluation statisitcs, rendered frames.
        '''
        if env is None:
            env = self.env
        if terminate_run_on_done is None:
            terminate_run_on_done = self.terminate_run_on_done

        self.x_prev = None
        self.u_prev = None
        if not env.initial_reset:
            env.set_cost_function_param(self.Q, self.R)
        # obs, info = env.reset()
        obs = env.reset()
        print('Init State:')
        print(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        self.setup_results_dict()
        self.results_dict['obs'].append(obs)
        self.results_dict['state'].append(env.state)
        i = 0
        if env.TASK == Task.STABILIZATION:
            if max_steps is None:
                MAX_STEPS = int(env.CTRL_FREQ * env.EPISODE_LEN_SEC)
            else:
                MAX_STEPS = max_steps
        elif env.TASK == Task.TRAJ_TRACKING:
            if max_steps is None:
                MAX_STEPS = self.traj.shape[1]
            else:
                MAX_STEPS = max_steps
        else:
            raise Exception('Undefined Task')
        self.terminate_loop = False
        done = False
        common_metric = 0
        while not (done and terminate_run_on_done) and i < MAX_STEPS and not (self.terminate_loop):
            action = self.select_action(obs)
            if self.terminate_loop:
                print('Infeasible MPC Problem')
                break
            # Repeat input for more efficient control.
            obs, reward, done, info = env.step(action)
            self.results_dict['obs'].append(obs)
            self.results_dict['reward'].append(reward)
            self.results_dict['done'].append(done)
            self.results_dict['info'].append(info)
            self.results_dict['action'].append(action)
            self.results_dict['state'].append(env.state)
            self.results_dict['state_mse'].append(info['mse'])
            # self.results_dict['state_error'].append(env.state - env.X_GOAL[i,:])

            common_metric += info['mse']
            print(i, '-th step.')
            print('action:', action)
            print('obs', obs)
            print('reward', reward)
            print('done', done)
            print(info)
            print()
            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            i += 1
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = '****** Evaluation ******\n'
            msg += 'eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n'.format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(),
                ep_returns.std())
        if len(frames) != 0:
            self.results_dict['frames'] = frames
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        self.results_dict['state'] = np.vstack(self.results_dict['state'])
        try:
            self.results_dict['reward'] = np.vstack(self.results_dict['reward'])
            self.results_dict['action'] = np.vstack(self.results_dict['action'])
            self.results_dict['full_traj_common_cost'] = common_metric
            self.results_dict['total_rmse_state_error'] = compute_state_rmse(self.results_dict['state'])
            self.results_dict['total_rmse_obs_error'] = compute_state_rmse(self.results_dict['obs'])
        except ValueError:
            raise Exception('[ERROR] mpc.run().py: MPC could not find a solution for the first step given the initial conditions. '
                            'Check to make sure initial conditions are feasible.')
        return deepcopy(self.results_dict)

    def reset_before_run(self, obs, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        # called once at the _evaluation_reset()
        self.reset()

    def wrap_sym(self, X):
        '''Wrap angle to [-pi, pi] when used in observation.

        Args:
            X (ndarray): The state to be wrapped.

        Returns:
            X_wrapped (ndarray): The wrapped state.
        '''
        X_wrapped = cs.fmod(X[0] + cs.pi, 2 * cs.pi) - cs.pi
        return X_wrapped
