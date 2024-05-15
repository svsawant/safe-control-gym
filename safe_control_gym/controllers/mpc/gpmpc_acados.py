


import time
from copy import deepcopy
from functools import partial

import casadi as cs
import gpytorch
import numpy as np
import scipy
import torch
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from skopt.sampler import Lhs

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import (GaussianProcessCollection, ZeroMeanIndependentGPModel,
                                                       covSEard, kmeans_centriods)
from safe_control_gym.controllers.mpc.linear_mpc import MPC, LinearMPC
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.gp_mpc import GPMPC
# from safe_control_gym.controllers.mpc.sqp_mpc import SQPMPC
from safe_control_gym.envs.benchmark_env import Task
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel

class GPMPC_ACADOS(GPMPC):
    '''Implements a GP-MPC controller with Acados optimization.'''
    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            constraint_tol: float = 1e-8,
            additional_constraints: list = None,
            soft_constraints: dict = None,
            warmstart: bool = True,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            n_ind_points: int = 30,
            inducing_point_selection_method = 'kmeans',
            recalc_inducing_points_at_every_step = False,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            online_learning: bool = False,
            prior_info: dict = None,
            sparse_gp: bool = False,
            # inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            **kwargs
    ):
        
        if prior_info is None or prior_info == {}:
            raise ValueError('GPMPC_ACADOS requires prior_prop to be defined. You may use the real mass properties and then use prior_param_coeff to modify them accordingly.')
        prior_info['prior_prop'].update((prop, val * prior_param_coeff) for prop, val in prior_info['prior_prop'].items())
        self.prior_env_func = partial(env_func, inertial_prop=prior_info['prior_prop'])
        if soft_constraints is None:
            self.soft_constraints_params = {'gp_soft_constraints': False,
                                            'gp_soft_constraints_coeff': 0,
                                            'prior_soft_constraints': False,
                                            'prior_soft_constraints_coeff': 0}
        else:
            self.soft_constraints_params = soft_constraints

        # print('self.soft_constraints_params:', self.soft_constraints_params)
        # print('self.soft_constraints_params["prior_soft_constraints"]:', \
        #       self.soft_constraints_params['prior_soft_constraints'])
        # Initialize the method using linear MPC.
        self.prior_ctrl = LinearMPC(
            self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
            terminate_run_on_done=terminate_run_on_done,
            prior_info=prior_info,
            # runner args
            # shared/base args
            output_dir=output_dir,
            additional_constraints=additional_constraints,
        )
        self.prior_ctrl.reset()
        self.sparse_gp = sparse_gp
        # self.prior_ctrl = SQPMPC(
        #     env_func = self.prior_env_func,
        #     seed= seed,
        #     horizon = horizon,
        #     q_mpc = q_mpc,
        #     r_mpc = r_mpc,
        #     warmstart= warmstart,
        #     soft_constraints= self.soft_constraints_params['prior_soft_constraints'],
        #     terminate_run_on_done= terminate_run_on_done,
        #     prior_info= prior_info,
        #     output_dir= output_dir,
        #     additional_constraints= additional_constraints)
        # self.prior_ctrl.reset()
        # super().__init__() # TODO: check the inheritance of the class
        super().__init__(
            env_func = env_func,
            seed= seed,
            horizon = horizon,
            q_mpc = q_mpc,
            r_mpc = r_mpc,
            constraint_tol = constraint_tol,
            additional_constraints = additional_constraints,
            soft_constraints = soft_constraints,
            warmstart = warmstart,
            train_iterations = train_iterations,
            test_data_ratio = test_data_ratio,
            overwrite_saved_data = overwrite_saved_data,
            optimization_iterations = optimization_iterations,
            learning_rate = learning_rate,
            normalize_training_data = normalize_training_data,
            use_gpu = use_gpu, 
            gp_model_path = gp_model_path,
            prob = prob,
            initial_rollout_std = initial_rollout_std,
            input_mask = input_mask,
            target_mask = target_mask,
            gp_approx = gp_approx,
            sparse_gp = sparse_gp,
            n_ind_points = n_ind_points,
            inducing_point_selection_method = 'kmeans',
            recalc_inducing_points_at_every_step = False,
            online_learning = online_learning,
            prior_info = prior_info,
            # inertial_prop: list = [1.0],
            prior_param_coeff = prior_param_coeff,
            terminate_run_on_done = terminate_run_on_done,
            output_dir = output_dir,
            **kwargs)
        # self.prior_ctrl = LinearMPC(
        #     self.prior_env_func,
        #     horizon=horizon,
        #     q_mpc=q_mpc,
        #     r_mpc=r_mpc,
        #     warmstart=warmstart,
        #     soft_constraints=self.soft_constraints_params['prior_soft_constraints'],
        #     terminate_run_on_done=terminate_run_on_done,
        #     prior_info=prior_info,
        #     # runner args
        #     # shared/base args
        #     output_dir=output_dir,
        #     additional_constraints=additional_constraints,
        # )
        # self.prior_ctrl.reset()
        # # Setup environments.
        self.env_func = env_func
        self.env = env_func(randomized_init=False, seed=seed)
        self.env_training = env_func(randomized_init=True, seed=seed)
        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.data_inputs = None
        self.data_targets = None
        self.prior_dynamics_func = self.prior_ctrl.linear_dynamics_func
        # self.prior_dynamics_func = self.prior_ctrl.dynamics_func # nonlinear prior 
        self.X_EQ = self.prior_ctrl.X_EQ
        self.U_EQ = self.prior_ctrl.U_EQ
        # GP and training parameters.
        self.gaussian_process = None
        self.train_iterations = train_iterations
        self.test_data_ratio = test_data_ratio
        self.overwrite_saved_data = overwrite_saved_data
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        self.normalize_training_data = normalize_training_data
        self.prob = prob
        if input_mask is None:
            self.input_mask = np.arange(self.model.nx + self.model.nu).tolist()
        else:
            self.input_mask = input_mask
        if target_mask is None:
            self.target_mask = np.arange(self.model.nx).tolist()
        else:
            self.target_mask = target_mask
        Bd = np.eye(self.model.nx)
        self.Bd = Bd[:, self.target_mask]
        self.gp_approx = gp_approx
        self.online_learning = online_learning
        self.last_obs = None
        self.last_action = None
        self.initial_rollout_std = initial_rollout_std
        # MPC params
        self.gp_soft_constraints = self.soft_constraints_params['gp_soft_constraints']
        self.gp_soft_constraints_coeff = self.soft_constraints_params['gp_soft_constraints_coeff']

        self.init_solver = 'ipopt'
        self.max_qp_iter = 50
        self.action_convergence_tol = 1e-5
        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        
        self.use_RTI = False
        self.setup_prior_dynamics()
        self.setup_acados_model()
        self.setup_acados_optimizer()
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp)
        # exit()
    
    # def set_lin_gp_dynamics_func(self):
    #     '''Updates symbolic dynamics with actual control frequency.'''
    #     # Original version, used in shooting.
    #     delta_x = cs.MX.sym('delta_x', self.model.nx, 1)
    #     delta_u = cs.MX.sym('delta_u', self.model.nu, 1)
    #     x_guess = cs.MX.sym('x_guess', self.model.nx, 1)
    #     u_guess = cs.MX.sym('u_guess', self.model.nu, 1)
    #     dfdxdfdu = self.model.df_func(x=x_guess, u=u_guess)
    #     dfdx = dfdxdfdu['dfdx']#.toarray()
    #     dfdu = dfdxdfdu['dfdu']#.toarray()
    #     z = cs.MX.sym('z', self.model.nx + self.model.nu, 1) # query point (the linearization point)
    #     Ad = cs.DM_eye(self.model.nx) + dfdx * self.dt
    #     Bd = dfdu * self.dt
    #     A_gp = self.gaussian_process.casadi_linearized_predict(z=z)['A']
    #     B_gp = self.gaussian_process.casadi_linearized_predict(z=z)['B']
    #     assert A_gp.shape == (self.model.nx, self.model.nx)
    #     assert B_gp.shape == (self.model.nx, self.model.nu)
    #     A = Ad + A_gp # TODO: check why Bd is used here correctly
    #     B = Bd + B_gp
    #     x_dot_lin = A @ delta_x + B @ delta_u
    #     self.linear_gp_dynamics_func = cs.Function('linear_dynamics_func', 
    #                                             [delta_x, delta_u, x_guess, u_guess, z], 
    #                                             [x_dot_lin, A, B],
    #                                             ['x0', 'p', 'x_guess', 'u_guess', 'z'],
    #                                             ['xf', 'A', 'B'])
    #     self.dfdx = A
    #     self.dfdu = B
    def setup_acados_model(self) -> AcadosModel:

        model_name = self.env.NAME
        
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = model_name

        A_lin = self.discrete_dfdx
        B_lin = self.discrete_dfdu

        if self.gaussian_process is None:
            f_disc = A_lin @ (acados_model.x - self.X_EQ) + B_lin @ (acados_model.u - self.U_EQ)
        else:
            f_disc = A_lin @ (acados_model.x - self.X_EQ) \
                + B_lin @ (acados_model.u - self.U_EQ) \
                + self.Bd @ self.gaussian_process.casadi_predict(z=cs.vertcat(acados_model.x, acados_model.u))['mean']
        
        acados_model.disc_dyn_expr = f_disc

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model

    def setup_acados_optimizer(self):
        print('=================Setting up acados optimizer=================')
        pass
        # before_optimizer_setup = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

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

        # T = self.T
        # if self.qp_solver in ['qrqp', 'qpoases']:
        #     opti = cs.Opti('conic')
        # else:
        #     opti = cs.Opti()
        # # States and inputs.
        # x_var = opti.variable(nx, T + 1)
        # u_var = opti.variable(nu, T)
        # # Linearization reference
        # x_guess = opti.parameter(nx, T + 1)
        # u_guess = opti.parameter(nu, T)
        # # Initial state.
        # x_init = opti.parameter(nx, 1)
        # # Reference trajectory (equilibrium point or trajectory, last step for terminal cost).
        # # essentially goal states
        # x_ref = opti.parameter(nx, T + 1)
        # # Add slack variables for soft constraints.
        # # if self.gp_soft_constraints:
        # #     state_slack_list = []
        # #     for state_constraint in self.constraints.state_constraints:
        # #         state_slack_list.append(opti.variable(state_constraint.num_constraints, T + 1))
        # #     input_slack_list = []
        # #     for input_constraint in self.constraints.input_constraints:
        # #         input_slack_list.append(opti.variable(input_constraint.num_constraints, T))
        # #     soft_con_coeff = self.gp_soft_constraints_coeff
        # # # Chance constraint limits.
        # # state_constraint_set = []
        # # for state_constraint in self.constraints.state_constraints:
        # #     state_constraint_set.append(state_constraint.constraint_set)
        # # input_constraint_set = []
        # # for input_constraint in self.constraints.input_constraints:
        # #     input_constraint_set.append(opti.parameter(input_constraint.num_constraints, T))



        # # Sparse GP mean postfactor matrix. (not used here!)
        # # TODO: check if this is needed
        # mean_post_factor = opti.parameter(len(self.target_mask), self.train_data['train_targets'].shape[0])
        
        # # cost (cumulative)
        # cost = 0
        # cost_func = self.model.loss
        # for i in range(T):
        #     cost += cost_func(x=x_var[:, i] + x_guess[:, i],
        #                       u=u_var[:, i] + u_guess[:, i],
        #                       Xr=x_ref[:, i],
        #                       Ur=np.zeros((nu, 1)),
        #                       Q=self.Q,
        #                       R=self.R)['l']
        # # Terminal cost.
        # cost += cost_func(x=x_var[:, -1] + x_guess[:, -1],
        #                   u=np.zeros((nu, 1)) + u_guess[:, -1],
        #                   Xr=x_ref[:, -1],
        #                   Ur=np.zeros((nu, 1)),
        #                   Q=self.Q,
        #                   R=self.R)['l']
        # # query point
        # # z = cs.vertcat(x_var[:, :-1]+x_guess[:,:-1], u_var+u_guess)
        # z = cs.vertcat(x_guess[:, :-1], u_guess)
        # z = z[self.input_mask, :]
        # # Constraints
        # for i in range(self.T):
        #     # Dynamics constraints using the dynamics of the prior and the mean of the GP.
        #     next_state = self.linear_gp_dynamics_func(x0=x_var[:, i], p=u_var[:, i], \
        #                                          x_guess=x_guess[:,i], u_guess=u_guess[:,i], \
        #                                          z=z[:, i])['xf']
        #     opti.subject_to(x_var[:, i + 1] == next_state)
        #     # TODO: probablistic constraints tightening
        #     for sc_i, state_constraint in enumerate(self.state_constraints_sym):
        #         opti.subject_to(state_constraint(x_var[:, i] + x_guess[:, i]) <= -self.constraint_tol)
                
        #     for ic_i, input_constraint in enumerate(self.input_constraints_sym):
        #         opti.subject_to(input_constraint(u_var[:, i] + u_guess[:, i]) <= -self.constraint_tol)

        # # final state constraint
        # for sc_i, state_constraint in enumerate(self.state_constraints_sym):
        #     opti.subject_to(state_constraint(x_var[:, -1] + x_guess[:, -1]) <= -self.constraint_tol)
        # # initial condiiton constraints
        # opti.subject_to(x_var[:, 0] + x_guess[:, 0] == x_init)
        # opti.minimize(cost)
        # # create solver 
        # opts = {'expand': True}
        # opti.solver(self.qp_solver, opts)
        # self.opti_dict = {
        #     'opti': opti,
        #     'x_var': x_var,
        #     'u_var': u_var,
        #     'x_guess_var': x_guess,
        #     'u_guess_var': u_guess,
        #     'x_init': x_init,
        #     'x_ref': x_ref,
        #     'cost': cost
        # }
        # after_optimizer_setup = time.time()
        # print('MPC setup_sqp_optimizer time: ', after_optimizer_setup - before_optimizer_setup)

    def select_action(self, obs, info=None):
        print('current obs:', obs)
        time_before = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            # set initial condition (0-th state)
            self.acados_ocp_solver.set(0, "lbx", obs)
            self.acados_ocp_solver.set(0, "ubx", obs)

            for idx in range(self.T + 1):
                init_x = np.zeros(nx)
                self.acados_ocp_solver.set(idx, "x", init_x)
            for idx in range(self.T):
                init_u = np.zeros(nu)
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
            status = self.acados_ocp_solver.solve()
            if status not in [0, 2]:
                self.acados_ocp_solver.print_statistics()
                raise Exception(f'acados returned status {status}. Exiting.')
                # print(f"acados returned status {status}. ")
            if status == 2:
                print(f"acados returned status {status}. ")
            action = self.acados_ocp_solver.get(0, "u")
        time_after = time.time()
        print('gpmpc acados action selection time:', time_after - time_before)
        return action
    
 
    def reset(self):
        '''Reset the controller before running.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        
        if self.gaussian_process is not None:
            self.set_gp_dynamics_func(self.n_ind_points)
        self.setup_acados_model()
        self.setup_acados_optimizer()
            # n_ind_points = self.train_data['train_targets'].shape[0]
        print('=========== Resetting prior controller ===========')
        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None

    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_next_seq
                                 ):
        '''Converts trajectory data for GP trianing.

        Args:
            x_seq (list): state sequence of np.array (nx,).
            u_seq (list): action sequence of np.array (nu,).
            x_next_seq (list): next state sequence of np.array (nx,).

        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).
        '''
        print("=========== Preprocessing training data ===========")
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        x_pred_seq = self.prior_dynamics_func(x0=x_seq.T,
                                              p=u_seq.T)['xf'].toarray()
        targets = (x_next_seq.T - x_pred_seq).transpose()  # (N, nx).
        inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).
        return inputs, targets
