from __future__ import annotations

import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Sequence

import casadi as cs
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.math_and_models.distributions import Normal
from safe_control_gym.math_and_models.neural_networks import MLP
from safe_control_gym.controllers.ppo_mpc.ppo_mpc_utils import (
    PPOBuffer,
    MLPCritic,
    MLPActorCritic,
    compute_returns_and_advantages,
)

# Phi utilities

def build_phi_future_mask(n_y: int, n_u: int, horizon: int) -> np.ndarray:
    """Build the boolean causal mask for Phi_future."""
    
    n_target = horizon * n_y
    n_future_u = horizon * n_u
    mask = np.zeros((n_target, n_future_u), dtype=bool)
    for i in range(horizon):          # output block index (step i)
        for j in range(i + 1):        # input block index (only causal: j <= i)
            mask[i * n_y:(i + 1) * n_y,
                 j * n_u:(j + 1) * n_u] = True
    return mask


def unwrap_phi(phi_past: np.ndarray,
               phi_future: np.ndarray,
               mask_future: np.ndarray) -> np.ndarray:
    """Extract all active Phi entries into a flat vector for use as theta.

    The RL optimizer tunes this vector.  Zero entries of Phi_future (non-causal)
    are never included and therefore never modified by RL.
    """
    
    past_vec = phi_past.reshape(-1)           # all n_target * n_past entries
    future_vec = phi_future[mask_future]       # only causal entries
    return np.concatenate([past_vec, future_vec]).astype(np.float64)


def wrap_phi_numpy(phi_active_vec: np.ndarray,
                   n_target: int,
                   n_past: int,
                   n_future_u: int,
                   mask_future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct Phi_past and Phi_future from a flat active-entry vector."""
    
    n_phi_past = n_target * n_past
    phi_past = phi_active_vec[:n_phi_past].reshape(n_target, n_past)
    phi_future = np.zeros((n_target, n_future_u), dtype=np.float64)
    phi_future[mask_future] = phi_active_vec[n_phi_past:]
    return phi_past, phi_future


def update_io_guess(Y_prev: np.ndarray,
                    U_prev: np.ndarray,
                    sigma_prev: np.ndarray,
                    n_y: int,
                    n_u: int) -> np.ndarray:
    """Produce a warm-start by shifting the previous QP solution by one step."""
    
    Y_guess = np.concatenate([Y_prev[n_y:], Y_prev[-n_y:]])
    U_guess = np.concatenate([U_prev[n_u:], U_prev[-n_u:]])
    sigma_guess = np.zeros_like(sigma_prev)
    return np.concatenate([Y_guess, U_guess, sigma_guess])


class PhiMPCFunction:
    def __init__(
        self,
        env,
        gamma: float,
        phi_past_init: np.ndarray,
        phi_future_init: np.ndarray,
        mask_future: np.ndarray,
        y_indices: Sequence[int],
        t_ini: int,
        theta_max: float,
        theta_index_in_y: int,
        horizon: int = 40,
        warmstart: bool = True,
        n_parallel_solver: int = 1,
        n_train_solver: int = 1,
        jit: bool = False,
        jit_options: dict = None,
        **kwargs,
    ):
        """Initialise the Phi-MPC QP and all CasADi sensitivity functions."""
        self.env = env
        self.gamma = gamma
        self.y_indices = list(y_indices)
        self.n_y = len(self.y_indices)
        self.t_ini = int(t_ini)
        self.horizon = int(horizon)
        self.theta_max = float(theta_max)
        self.theta_index_in_y = int(theta_index_in_y)
        self.warmstart = warmstart
        self.n_parallel_solver = int(n_parallel_solver)
        self.n_train_solver = int(n_train_solver)
        self.jit = jit
        self.jit_options = jit_options if jit_options is not None else {}

        # Derive dimensions
        self.n_u = int(np.asarray(env.action_space.shape).prod())
        self.n_past = self.t_ini * self.n_u + (self.t_ini + 1) * self.n_y
        self.n_target = self.horizon * self.n_y
        self.n_future_u = self.horizon * self.n_u

        # Store initial Phi matrices and mask for wrap/unwrap
        self.phi_past_init = phi_past_init.astype(np.float64)
        self.phi_future_init = phi_future_init.astype(np.float64)
        self.mask_future = mask_future

        # Derive number of active Phi entries in theta
        self.n_phi_past_active = self.n_target * self.n_past 
        self.n_phi_future_active = int(mask_future.sum())
        self.n_phi_active = self.n_phi_past_active + self.n_phi_future_active

        # Set action space bounds
        self.u_lo = np.asarray(env.physical_action_bounds[0], dtype=np.float64).reshape(-1)
        self.u_hi = np.asarray(env.physical_action_bounds[1], dtype=np.float64).reshape(-1)

        
        # TODO: this is stale, for consistency I should get the u_eq from my env wrapper
        # This does not hurt, but looks weird. 
        assert hasattr(env, "symbolic") and hasattr(env.symbolic, "U_EQ")
        self.u_eq = np.asarray(env.symbolic.U_EQ, dtype=np.float64).reshape(-1)

        # Equilibrium output (used to initialise I/O history buffers)
        assert hasattr(env, "X_GOAL"), "PhiMPCFunction requires env.X_GOAL"
        x_goal = np.asarray(env.X_GOAL, dtype=np.float64)
        target = x_goal[0] if x_goal.ndim > 1 else x_goal # stab. vs. tracking
        self.y_eq = target[self.y_indices]

        # Task mode and reference trajectory
        if env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = env.X_GOAL
            # traj: (n_y, 1) repeated — used by get_y_ref for stabilization
            self.traj = np.tile(self.y_eq.reshape(-1, 1), (1, self.horizon + 1))
            self.traj_step = 0
        elif env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            x_goal = np.asarray(env.X_GOAL, dtype=np.float64)
            self.traj = x_goal[:, self.y_indices].T
            self.traj_step = 0

        # Per-env warm-start buffers for batch rollout.
        self.io_hists = [None] * self.n_parallel_solver

        # Build the QP and all CasADi functions 
        self.solver_dict = None  # populated by setup_optimizer()
        self.setup_optimizer()

    def setup_optimizer(self):
        """Build the CasADi QP and all KKT sensitivity functions.
        """
        
        horizon = self.horizon
        n_y = self.n_y
        n_u = self.n_u
        n_target = self.n_target
        n_past = self.n_past
        n_future_u = self.n_future_u
        n_phi_past_active = self.n_phi_past_active
        n_phi_future_active = self.n_phi_future_active
        n_sigma = horizon  # one slack per timestep for soft theta constraint

        # NB! The soft theta constraint is silly, just something I tried. TODO: remove!
        start_time = time.time()

        # Define decision variables
        Y_var = cs.MX.sym("Y", n_target)
        U_var = cs.MX.sym("U", n_future_u)
        sigma_var = cs.MX.sym("sigma", n_sigma)
        opt_vars = cs.vertcat(Y_var, U_var, sigma_var)

        # Function for concatenating the decision variables [Y_var, U_var, sigma_var] into a single solver vector opt_vars
        opt_vars_fn = cs.Function("opt_vars_fn", [Y_var, U_var, sigma_var], [opt_vars])
        
        # Opposite function for splitting the vector opt_vars back into decision variables [Y_var, U_var, sigma_var]
        yus_fn = cs.Function("yus_fn", [opt_vars], [Y_var, U_var, sigma_var])
        
        # Function for extracting first control input u_t from the optimized variable vector
        opt_act_fn = cs.Function("opt_act_fn", [opt_vars], [U_var[:n_u]])

        # PARAMETERS
        #
        # Fixed parameters (change every step, NOT learned by RL):
        #   z_past_p: I/O history for the current timestep, shape (n_past,)
        #
        # Reference parameters (change every step, NOT learned by RL):
        #   y_ref_p: stacked output reference over horizon, shape (n_target,)
        #
        # Learnable parameters (theta, tuned by RL via KKT sensitivity):
        #   Qy_sym:          output cost weights, shape (n_y,)
        #   R_sym:           input cost weights, shape (n_u,)
        #   tw_sym:          terminal weight scale, scalar
        #   phi_active_sym:  active Phi entries, shape (n_phi_active,)
        #                    layout: [phi_past.flatten(), phi_future[mask_future]]
        
        # Fixed params        
        z_past_p = cs.MX.sym("z_past", n_past)
        y_ref_p = cs.MX.sym("y_ref", n_target)

        # Learnable params
        Qy_sym = cs.MX.sym("Qy", n_y)
        R_sym = cs.MX.sym("R_cost", n_u)
        tw_sym = cs.MX.sym("tw", 1) # terminal weight, scalar
        ws_sym = cs.MX.sym("ws", 1) # slack penalty weight, silly
        tm_sym = cs.MX.sym("tm", 1) # theta max bound, silly
        phi_active_sym = cs.MX.sym("phi_active", self.n_phi_active)

        theta_param = cs.vertcat(Qy_sym, R_sym, tw_sym, ws_sym, tm_sym, phi_active_sym)

        # Symbolic phi reconstruction
        
        # Important! 
        # CasADi cs.reshape uses Fortran (column-major) order, but phi_past_vec
        # was flattened in C (row-major) order by numpy.  To recover the correct
        # (n_target, n_past) matrix we reshape as (n_past, n_target) first and
        # then transpose.
        # TODO: consider changing this and treat the cause rather than the symptom? 
        phi_past_vec = phi_active_sym[:n_phi_past_active]
        phi_future_vec = phi_active_sym[n_phi_past_active:]
        
        Phi_past_sym = cs.reshape(phi_past_vec, n_past, n_target).T

        # Start with an all-zero matrix; fill in causal entries one by one.
        Phi_future_sym = cs.MX.zeros(n_target, n_future_u)
        row_idx, col_idx = np.where(self.mask_future)
        for k in range(len(row_idx)):
            # Element assignment in CasADi MX: sets the (r,c) entry symbolically.
            Phi_future_sym[int(row_idx[k]), int(col_idx[k])] = phi_future_vec[k]

        
        # Define the predictor equality constraints
        Y_pred = Phi_past_sym @ z_past_p + Phi_future_sym @ U_var
        H_eq = Y_var - Y_pred # later forced to zero
        
        # Build horizon-stacked stage/terminal weights and the repeated input reference
        Qy_blocks = [Qy_sym] * (horizon - 1) + [tw_sym * Qy_sym]
        Qy_full = cs.vertcat(*Qy_blocks)  # (n_target,)
        R_full = cs.vertcat(*([R_sym] * horizon))
        u_eq_stack = cs.DM(np.tile(self.u_eq, horizon))  # constant

        Y_err = Y_var - y_ref_p
        U_err = U_var - u_eq_stack

        # Assemble the cost function
        cost = (
            cs.dot(Qy_full, Y_err * Y_err) # output tracking cost
            + cs.dot(R_full, U_err * U_err) # input effort cost
            + ws_sym * cs.sum1(sigma_var) # soft constraint penalty (learnable) (silly)
        )

        # Constraints
        
        # Bookkeeping
        con_list, con_lbg, con_ubg, con_eq = [], [], [], []
        H_eq_sym_list = [] 
        H_ieq_sym_list = []

        # Equality: predictor constraint
        # Y == Phi_past @ z_past + Phi_future @ U + b rearranged as: H_eq = Y - Y_pred = 0
        con_list.append(H_eq)
        con_lbg.append(cs.DM.zeros(n_target, 1))
        con_ubg.append(cs.DM.zeros(n_target, 1))
        con_eq += [True] * n_target
        H_eq_sym_list.append(H_eq)

        # Hard input lower bounds
        u_lo_stack = cs.DM(np.tile(self.u_lo, horizon))
        u_hi_stack = cs.DM(np.tile(self.u_hi, horizon))
        ineq_ulo = u_lo_stack - U_var
        con_list.append(ineq_ulo)
        con_lbg.append(-cs.DM.inf(n_future_u, 1))
        con_ubg.append(cs.DM.zeros(n_future_u, 1))
        con_eq += [False] * n_future_u
        H_ieq_sym_list.append(ineq_ulo)

        # Hard input upper bounds
        ineq_uhi = U_var - u_hi_stack
        con_list.append(ineq_uhi)
        con_lbg.append(-cs.DM.inf(n_future_u, 1))
        con_ubg.append(cs.DM.zeros(n_future_u, 1))
        con_eq += [False] * n_future_u
        H_ieq_sym_list.append(ineq_uhi)

        # Soft pitch angle upper bound (silly)
        theta_idx = [k * n_y + self.theta_index_in_y for k in range(horizon)]
        Y_theta = Y_var[theta_idx]

        ineq_theta_hi = Y_theta - sigma_var - tm_sym
        con_list.append(ineq_theta_hi)
        con_lbg.append(-cs.DM.inf(horizon, 1))
        con_ubg.append(cs.DM.zeros(horizon, 1))
        con_eq += [False] * horizon
        H_ieq_sym_list.append(ineq_theta_hi)

        # Soft pitch angle lower bound (silly)
        ineq_theta_lo = -Y_theta - sigma_var - tm_sym
        con_list.append(ineq_theta_lo)
        con_lbg.append(-cs.DM.inf(horizon, 1))
        con_ubg.append(cs.DM.zeros(horizon, 1))
        con_eq += [False] * horizon
        H_ieq_sym_list.append(ineq_theta_lo)

        # -- sigma >= 0: -sigma <= 0
        ineq_sigma = -sigma_var
        con_list.append(ineq_sigma)
        con_lbg.append(-cs.DM.inf(horizon, 1))
        con_ubg.append(cs.DM.zeros(horizon, 1))
        con_eq += [False] * horizon
        H_ieq_sym_list.append(ineq_sigma)
        
        # TODO: yes, remove the soft constraint on theta, it really messes things up here and does not do anything. 

        # Concatenate all constraints into flat vectors
        con_list = cs.vcat(con_list)
        con_lbg = cs.vcat(con_lbg)
        con_ubg = cs.vcat(con_ubg)
        H_eq_sym = cs.vcat(H_eq_sym_list)
        H_ieq_sym = cs.vcat(H_ieq_sym_list)

        # QP solver
        opts_setting = {
            "print_time": 0, # silent
            "error_on_fail": False, # do not immediately crash on solver failure
            "printLevel": "none", # silent (but it does not listen to me :-()
        }

        qp_prob = {
            "f": cost,
            "x": opt_vars,
            "p": cs.vertcat(z_past_p, y_ref_p, theta_param),
            "g": con_list,
        }
        pisolver = cs.qpsol("pisolver", "qpoases", qp_prob, opts_setting)
        print(f"[PhiMPC Setup] QP problem setup time: {time.time() - start_time:.3f} s.")
        start_time = time.time()

        # KKT SENSITIVITY SETUP
        #
        # After solving for z* = [Y*, U*, sigma*, lambda*, mu*], we need
        # the sensitivity du*_0/dtheta for the RL gradient.
        #
        # Method (identical to MPCFunction):
        #   1. Build the Lagrangian L = cost + lambda'*H_eq + mu'*H_ieq
        #   2. KKT residual R_kkt = [grad_w L, H_eq, mu * H_ieq + epsilon]
        #   3. z_kkt = [opt_vars, mult]
        #   4. Compute dR/dz, dR/dtheta symbolically via CasADi .factory()
        #   5. Solve: dPi_prime = S @ (dR/dz)^{-1}  where S selects u*_0
        #   6. dPi = -(dPi_prime @ dR/dtheta).T  [shape (n_theta, n_u)]
        #
        # Because Phi appears linearly in H_eq (not in the Hessian of cost),
        # dR/d(Phi_entry) is simple and well-conditioned.
         
        
        etau = 1e-5 # small perturbation for perturbed complementarity in KKT residual
        n_eq = n_target
        n_ineq = 2 * n_future_u + 3 * horizon
        lamb = cs.MX.sym("lamb", n_eq) # equality multipliers (lambda)
        mu = cs.MX.sym("mu", n_ineq) # inequality multipliers
        mult = cs.vertcat(lamb, mu)

        # Lagrangian: L = cost + lambda' H_eq + mu' H_ieq
        lagrangian = cost + cs.dot(lamb, H_eq_sym) + cs.dot(mu, H_ieq_sym)
        dlag_dw = cs.jacobian(lagrangian, opt_vars)  # (1, n_opt_vars)

        # KKT residual stacked as in MPCFunction: [stationarity, primal feasibility, perturbed complementarity]
        R_kkt = cs.vertcat(
            cs.transpose(dlag_dw), # stationarity: grad_w L = 0
            H_eq_sym, # primal feasibility (equality)
            mu * H_ieq_sym + etau, # complementarity (perturbed)
        )

        z_kkt = cs.vertcat(opt_vars, mult)  # full primal-dual vector

        # KKT functions for norm-checking and sensitivity evaluation.
        rkkt_fn = cs.Function(
            "rkkt_fn", [z_kkt, z_past_p, y_ref_p, theta_param], [R_kkt]
        )
        rkkt_norm_fn = cs.Function(
            "rkkt_norm_fn", [z_kkt, z_past_p, y_ref_p, theta_param],
            [cs.norm_2(R_kkt)]
        )

        # Use CasADi .factory() to compute the Jacobians of R_kkt
        dR_sensfunc = rkkt_fn.factory(
            "dR",
            ["i0", "i1", "i2", "i3"],
            ["jac:o0:i0", "jac:o0:i2", "jac:o0:i3"],
        )
        [dRdz, dRdP_ref, dRdP_theta] = dR_sensfunc(
            z_kkt, z_past_p, y_ref_p, theta_param
        )
        # We only need dRdP_theta for RL gradient (not dRdP_ref).
        dRdP = cs.horzcat(dRdP_theta)

        print(f"[PhiMPC Setup] KKT matrix setup time: {time.time() - start_time:.3f} s.")
        start_time = time.time()

        # Selection matrix S: extracts first-step input u*_0 from the
        # primal vector.  opt_vars layout: [Y (n_target), U (n_future_u), sigma]
        # so u*_0 occupies indices [n_target : n_target + n_u].
        S = cs.DM.zeros(n_u, dRdz.shape[0])
        for i in range(n_u):
            S[i, n_target + i] = 1.0  # select u_0[i] from opt_vars block

        # Compute dPi = sensitivity of u*_0 w.r.t. theta.
        # dPi_prime = S @ (dRdz)^{-1}  solved via:  dRdz.T @ X = S.T → X = (dRdz.T)^{-1}@S.T
        # Then dPi_prime = X.T = S @ dRdz^{-1}
        dPi_prime = cs.solve(dRdz.T, S.T).T
        dPi = -(dPi_prime @ dRdP).T  # shape (n_theta, n_u)

        # Return zeros when QP is not solved to optimality.
        dPi_zeros = cs.MX.zeros(dPi.shape)
        f_true = cs.Function("f_true", [z_kkt, z_past_p, y_ref_p, theta_param], [dPi])
        f_false = cs.Function("f_false", [z_kkt, z_past_p, y_ref_p, theta_param], [dPi_zeros])
        dPi_fn = cs.Function.if_else("dPi_fn", f_true, f_false)

        # Combined function: returns (KKT norm, dPi) in one call.
        all_fn = cs.Function(
            "all_fn",
            [z_kkt, z_past_p, y_ref_p, theta_param],
            [cs.norm_2(R_kkt), dPi],
        )
        print(f"[PhiMPC Setup] Sensitivity setup time: {time.time() - start_time:.3f} s.")
        start_time = time.time()

        # Trigger JIT compilation if specified
        if self.jit:
            _ = all_fn(
                cs.DM.zeros(z_kkt.shape),
                cs.DM.zeros(z_past_p.shape),
                cs.DM.zeros(y_ref_p.shape),
                cs.DM.zeros(theta_param.shape),
            )
            print(f"[PhiMPC Setup] JIT compilation time: {time.time() - start_time:.3f} s.")

        # Store everything needed for solving and sensitivity evaluation.
        self.solver_dict = {
            "Y_var": Y_var,
            "U_var": U_var,
            "sigma_var": sigma_var,
            "opt_vars": opt_vars,
            "opt_vars_fn": opt_vars_fn,
            "yus_fn": yus_fn,
            "opt_act_fn": opt_act_fn,
            "cost": cost,
            "lower_bound": con_lbg,
            "upper_bound": con_ubg,
            "solver": pisolver,
            "rkkt_fn": rkkt_fn,
            "rkkt_norm_fn": rkkt_norm_fn,
            "dpi_fn": dPi_fn,
            "all_fn": all_fn,
        }

    def reset(self, idx: int | None = None):
        """Clear I/O history buffers at the start of a new episode.
        """
        if idx is None:
            self.traj_step = 0
            self._Y_prev = self._U_prev = self._sigma_prev = None
        else:
            self.io_hists[idx] = None

    def get_y_ref(self, traj_step: int, traj_ref: np.ndarray | None = None) -> np.ndarray:
        """Construct the output reference vector y_ref for the current step.
        """
        if self.mode == "stabilization":
            return np.tile(self.y_eq, self.horizon)  # (n_target,)

        # Tracking: extract a window of length horizon from the trajectory.
        # QP predicts y_{t+1}...y_{t+N}, so reference window starts at t+1.
        traj = traj_ref if traj_ref is not None else self.traj  # (n_y, T)
        T = traj.shape[1]
        start = min(traj_step + 1, T - 1)
        end = min(traj_step + 1 + self.horizon, T)
        remain = self.horizon - (end - start)
        slices = [traj[:, start:end]]
        if remain > 0:
            slices.append(np.tile(traj[:, -1:], (1, remain)))
        y_ref_2d = np.concatenate(slices, axis=1)  # (n_y, horizon)
        return y_ref_2d.T.reshape(-1)               # (n_target,)

    def select_action(self, obs: np.ndarray, theta: np.ndarray,
                      traj_ref, info=None, mode="eval", z_past: np.ndarray = None):
        """Solve the QP for a single environment and return the first-step action.
        """

        # Fetch helper objects built earlier in setup_optimizer and z_past
        solver = self.solver_dict["solver"]
        yus_fn = self.solver_dict["yus_fn"]
        z_past = np.asarray(z_past, dtype=np.float64)
        
        # TODO: this is clunky. Fix? 
        traj_step = self.traj_step
        if traj_ref is not None:
            y_ref = np.asarray(traj_ref[0], dtype=np.float64)
        else:
            y_ref = self.get_y_ref(traj_step, None)

        # If tracking, look to the next point. If stabilizing, all points are the same. 
        if self.mode == "tracking":
            self.traj_step += 1

        # Build full parameter vector for the solver
        p_param = np.concatenate([z_past, y_ref, theta])
        
        # Prepare the initial guess for the optimization variables
        n_opt = self.solver_dict["opt_vars"].shape[0]
        opt_vars_init = np.zeros(n_opt)
        if self.warmstart and hasattr(self, "_Y_prev") and self._Y_prev is not None:
            opt_vars_init = update_io_guess(
                self._Y_prev, self._U_prev, self._sigma_prev, self.n_y, self.n_u
            )

        # Solve
        soln = solver(
            x0=opt_vars_init,
            p=p_param,
            lbg=self.solver_dict["lower_bound"],
            ubg=self.solver_dict["upper_bound"],
        )
        optimal = solver.stats()["success"]
        if not optimal:
            print(f"[PhiMPC] OSQP solve failed (status: {solver.stats().get('return_status','?')}); "
                  "using best-effort solution.")

        # Extract solution
        opt_vars_val = soln["x"].full()
        Y_val, U_val, sigma_val = yus_fn(opt_vars_val)
        Y_val = Y_val.full()
        U_val = U_val.full()
        sigma_val = sigma_val.full().flatten()

        # Store flat for warm-starting
        self._Y_prev = Y_val.flatten()
        self._U_prev = U_val.flatten()
        self._sigma_prev = sigma_val

        # First-step action
        U_flat = U_val.flatten()
        action = np.clip(U_flat[:self.n_u], self.u_lo, self.u_hi)

        # Update I/O history with action that will be applied
        # TODO: t_wall is unused
        results_dict = {
            # z_past stored here so select_action_batch_train can retrieve it.
            "z_past": deepcopy(z_past),
            "y_ref": deepcopy(y_ref),
            "horizon_outputs": deepcopy(Y_val),
            "horizon_inputs": deepcopy(U_val),
            "t_wall": solver.stats()["t_wall_solver"],
        }
        solver_info = {
            "success": optimal,
            "opt_var": opt_vars_val[:, 0] if opt_vars_val.ndim > 1 else opt_vars_val.flatten(),
            "fixed_param": deepcopy(z_past),
            "ref_param": deepcopy(y_ref),
            "theta_param": deepcopy(theta),
            "traj_step": traj_step,
        }
        return action, solver_info, results_dict, optimal

    def get_parallel_solver(self, n_solvers: int):
        """Create parallelised (mapped) solver instances.
        """
        pi_solvers = self.solver_dict["solver"].map(n_solvers, "thread")
        rkkt_norm_fns = self.solver_dict["rkkt_norm_fn"].map(n_solvers, "thread")
        dpi_solvers = self.solver_dict["dpi_fn"].map(n_solvers, "thread")
        all_solvers = self.solver_dict["all_fn"].map(n_solvers, "thread")
        return pi_solvers, rkkt_norm_fns, dpi_solvers, all_solvers



class PhiMPCPolicyFunction(PhiMPCFunction):
    """Extends PhiMPCFunction with batched rollout and training-update methods.
    """

    def __init__(self, *args, **kwargs):
        # Intializes the class from which it inherits, PhiMPCFunction
        super().__init__(*args, **kwargs)
        
        # Build parallel solver pools for rollout collection and training
        self.pi_solvers, self.rkkt_norm_fns, _, _ = self.get_parallel_solver(
            self.n_parallel_solver
        )
        self.pi_solvers_train, _, _, self.all_solvers_train = self.get_parallel_solver(
            self.n_train_solver
        )

    def select_action_batch(self,
                            obs_batch: np.ndarray,
                            theta: np.ndarray,
                            traj_ref_batch: list,
                            actor_info: list,
                            z_past_batch: np.ndarray = None) -> tuple:
        """Solve QPs for N environments in parallel during rollout collection.
        """
        
        # Safety check, uneccessary bloat? TODO remove? 
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        
        # How many batches? 
        N = obs_batch.shape[0]

        # Fetch lower and upper bounds for constraint vectors
        con_lbg = self.solver_dict["lower_bound"]
        con_ubg = self.solver_dict["upper_bound"]
        
        # Helper function to unpack solver decision vector
        yus_fn = self.solver_dict["yus_fn"]
        
        # Helper function to assemble optimization variables (TODO: unused)
        opt_vars_fn = self.solver_dict["opt_vars_fn"]
        
        # How many optimization variables in one QP-solve? 
        n_opt = self.solver_dict["opt_vars"].shape[0]

        # Collect per-env parameters and prepare containers
        x0_list, fixed_p_list, ref_p_list = [], [], []
        lbg = con_lbg.full().repeat(N, 1)
        ubg = con_ubg.full().repeat(N, 1)

        # For all envs...
        for i in range(N):
            # Save input-output history and initialize warm-start memory (if does not already exist)
            z_past = np.asarray(z_past_batch[i], dtype=np.float64)

            if self.io_hists[i] is None:
                self.io_hists[i] = {"_Y_prev": None, "_U_prev": None,
                                    "_sigma_prev": None}

            # Build trajectory reference to be tracked
            traj_step = int(actor_info[i]["current_step"])
            x_ref = np.asarray(actor_info[i]["x_ref"], dtype=np.float64).T  # (nx, T)
            y_ref = self.get_y_ref(traj_step, x_ref[self.y_indices, :])      # (n_target,)

            # Warm-start: shift previous solution if available (not at episode start).
            opt_vars_init = np.zeros(n_opt)
            if (self.warmstart
                    and self.io_hists[i]["_Y_prev"] is not None
                    and traj_step > 0):
                opt_vars_init = update_io_guess(
                    self.io_hists[i]["_Y_prev"],
                    self.io_hists[i]["_U_prev"],
                    self.io_hists[i]["_sigma_prev"],
                    self.n_y, self.n_u,
                )

            # Initial guess, fixed input-output history, trajectory reference
            x0_list.append(opt_vars_init)
            fixed_p_list.append(z_past)
            ref_p_list.append(y_ref)

        # Get everything on CasADi-friendly form
        x0 = np.array(x0_list).T
        fixed_p = np.array(fixed_p_list).T
        ref_p = np.array(ref_p_list).T
        p = np.concatenate([fixed_p, ref_p, theta.T], axis=0)

        # Solve all N QPs in parallel
        soln_batch = self.pi_solvers(
            x0=x0, p=p, lbg=lbg, ubg=ubg
        )
        
        # Build full KKT state for each solve
        z_kkt = cs.vertcat(soln_batch["x"], soln_batch["lam_g"])
        
        # Compute residual KKT norm for each batch
        rkkt_norm_batch = self.rkkt_norm_fns(z_kkt, fixed_p, ref_p, theta.T)
        
        # Convert to NumPy
        rkkt_norms = rkkt_norm_batch.full()
        optimal_batch = rkkt_norms < 1e-3  # matches original PPO-MPC threshold

        # Post-processing - extract actions and update histories
        action_batch, results_dict_batch, info_batch = [], [], []
        
        # For every env in batch...
        for i, obs in enumerate(obs_batch):
            # Extract optimal solution and unpack
            opt_vars_val = soln_batch["x"].full()[:, i]
            Y_val, U_val, sigma_val = yus_fn(opt_vars_val)
            Y_val = Y_val.full().flatten()
            U_val = U_val.full().flatten()
            sigma_val = sigma_val.full().flatten()

            # Store flat for warm-starting
            self.io_hists[i]["_Y_prev"] = Y_val
            self.io_hists[i]["_U_prev"] = U_val
            self.io_hists[i]["_sigma_prev"] = sigma_val

            # Extract first planned control move as action
            action = np.clip(U_val[:self.n_u], self.u_lo, self.u_hi)
            
            if not optimal_batch[0, i]:
                # OSQP failed for this instance — fall back to hover to avoid a crash.
                print(f"[PhiMPC-rollout] env {i}: KKT norm={rkkt_norms[0,i]:.3e} >= 1e3 — "
                      "falling back to u_eq.", flush=True)
                action = self.u_eq.copy()

            results_dict = {
                # Store z_past for the PPO train pass; it cannot be reconstructed from the current obs alone
                "z_past": deepcopy(fixed_p_list[i]),
                "y_ref": deepcopy(ref_p_list[i]),
                "horizon_outputs": deepcopy(Y_val),
                "horizon_inputs": deepcopy(U_val),
            }
            solver_info = {
                "success": optimal_batch[0, i],
                "opt_var": opt_vars_val,          # warm-start for train pass
                "fixed_param": deepcopy(fixed_p_list[i]), # = z_past
                "ref_param": deepcopy(ref_p_list[i]), # = y_ref
                "theta_param": deepcopy(theta[i, :]),
                "traj_step": actor_info[i]["current_step"],
                "x_ref": actor_info[i]["x_ref"],
            }
            action_batch.append(action)
            results_dict_batch.append(results_dict)
            info_batch.append(solver_info)

        return action_batch, info_batch, results_dict_batch, optimal_batch

    def select_action_batch_train(self,
                                  obs_batch: np.ndarray,
                                  theta: np.ndarray,
                                  traj_ref,
                                  info_batch: list) -> tuple:
        """Solve QPs and compute KKT sensitivity for the PPO gradient update.
        """
        if not obs_batch.ndim > 1:
            obs_batch = obs_batch[None, :]
        N = obs_batch.shape[0]

        con_lbg = self.solver_dict["lower_bound"]
        con_ubg = self.solver_dict["upper_bound"]
        opt_act_fn = self.solver_dict["opt_act_fn"]
        n_opt = self.solver_dict["opt_vars"].shape[0]

        x0_list, fixed_p_list, ref_p_list = [], [], []
        lbg = con_lbg.full().repeat(N, 1)
        ubg = con_ubg.full().repeat(N, 1)

        for i in range(N):
            info = info_batch[i]
            # Retrieve the stored warm-start solution from rollout collection.
            opt_vars_init = info["opt_var"]
            # z_past and y_ref stored during select_action_batch().
            z_past = info["fixed_param"]   # (n_past,)
            y_ref = info["ref_param"]      # (n_target,)

            x0_list.append(opt_vars_init)
            fixed_p_list.append(z_past)
            ref_p_list.append(y_ref)

        x0 = np.array(x0_list).T
        fixed_p = np.array(fixed_p_list).T
        ref_p = np.array(ref_p_list).T
        p = np.concatenate([fixed_p, ref_p, theta.T], axis=0)

        # Solve for N QPs in parallel using the training solver pool
        soln_batch = self.pi_solvers_train(x0=x0, p=p, lbg=lbg, ubg=ubg)
        z_kkt = cs.vertcat(soln_batch["x"], soln_batch["lam_g"])
        action_batch = opt_act_fn(soln_batch["x"]).full().T  # (N, n_u)

        # Compute KKT sensitivity dPi = du*_0/dtheta for each env
        rkkt_norm_batch, dpi_cs = self.all_solvers_train(z_kkt, fixed_p, ref_p, theta.T)
        rkkt_norms_train = rkkt_norm_batch.full()  # shape (1, N)
        
        optimal_batch = rkkt_norms_train < 1e-3  # matches original PPO-MPC threshold
        # Store for aggregation in update() — avoids per-mini-batch print spam
        self._last_kkt_norms = rkkt_norms_train

        nabla_pi_theta_batch = []
        nabla_pi_ref_batch = []
        for i in range(N):
            nabla_pi_theta_batch.append(
                int(optimal_batch[0, i])  # zero gradient if not optimal
                * dpi_cs[:, self.n_u * i: self.n_u * (i + 1)].T.full()
            )
            
        # Convert to torch tensors to match PPO_MPC_Agent.compute_policy_loss() interface
        action_batch = torch.FloatTensor(action_batch)
        nabla_pi_ref_batch = torch.FloatTensor(np.zeros((N, self.n_u, 1)))  # unused
        nabla_pi_theta_batch = torch.FloatTensor(np.array(nabla_pi_theta_batch))
        optimal_batch = torch.FloatTensor(np.array(optimal_batch)).T
        return action_batch, nabla_pi_ref_batch, nabla_pi_theta_batch, optimal_batch



class PhiMPCActor(nn.Module):
    """Actor that maps observations to actions via the Phi-MPC QP."""
        
    def __init__(
        self,
        env,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple,
        activation: str,
        gamma: float,
        phi_past_init: np.ndarray,
        phi_future_init: np.ndarray,
        mask_future: np.ndarray,
        exploration_init: float,
        actor_config: dict,
    ):

        # We initialize the parent class of this class, i.e. the nn.Module
        super().__init__()

        # Build the QP solver
        # This is the object that defines the QP solved at each control step:
        # it stores the predictor, builds the optimization problem, and later
        # returns the first control action from the optimized input sequence.
        
        self.mpc = PhiMPCPolicyFunction(
            env, gamma,
            phi_past_init, phi_future_init, mask_future,
            **actor_config["mpc_config"],
        )

        # Initialise mpc_param
        qy_init = np.asarray(actor_config["qy_mpc"], dtype=np.float64)   # (n_y,)
        r_init = np.asarray(actor_config["r_mpc"], dtype=np.float64)      # (n_u,)
        tw_init = np.array([actor_config["terminal_weight_scale"]], dtype=np.float64)
        ws_init = np.array([actor_config.get("w_sigma_init", 10.0)], dtype=np.float64)
        tm_init = np.array([actor_config.get("theta_max_init", 0.8)], dtype=np.float64)
        phi_active_init = unwrap_phi(phi_past_init, phi_future_init, mask_future)

        # Split learnable parameters into two nn.Parameters so that Adam can use different learning rates 
        # Layout: [Qy, R, tw, ws, tm] — must match theta_param in CasADi QP.
        self.n_cost_params = len(qy_init) + len(r_init) + 3  # Qy + R + tw + ws + tm
        self.cost_param = nn.Parameter(
            torch.FloatTensor(np.concatenate([qy_init, r_init, tw_init, ws_init, tm_init]))
        )
        self.phi_param = nn.Parameter(torch.FloatTensor(phi_active_init))

        # Define the actors MLP called param_net that allows it to learn state-dependent parameters. 
        # (not sure if this works yet)
        n_theta = self.n_cost_params + len(phi_active_init)
        self.param_net = MLP(obs_dim, n_theta, list(hidden_dims), activation)

        # y_indices needed for get_references() to extract outputs from x_ref.
        self.y_indices = self.mpc.y_indices

        # Action distribution
        # Gaussian: mean = QP optimal action, std = exp(logstd).
        # logstd is learnable but starts small (near-deterministic).
        self.logstd = nn.Parameter(exploration_init * torch.ones(act_dim))
        self.dist_fn = lambda x: Normal(x, self.logstd.exp())
        # I think saving this here is unnecessary for my approach, I think its old. TODO: delete?  
        self.traj = self.mpc.traj

    @property
    def mpc_param(self):
        """Convenience: concatenated [cost_param, phi_param] (non-leaf)."""
        return torch.cat([self.cost_param, self.phi_param])

    def get_theta_param(self, obs):
        """Return the current theta = [cost_param, phi_param] (broadcast to batch if needed).

        The param_net output is multiplied by 0.0 so the result is effectively
        constant (state-independent), matching the original MPCActor design.
        A small jitter is added to help the QP solver avoid degenerate cases.
        """
        theta_base = torch.cat([self.cost_param, self.phi_param])
        if obs.ndim > 1:
            theta = (theta_base.repeat(obs.shape[0], 1)
                     + 0.0 * self.param_net.forward(torch.FloatTensor(obs)))
        else:
            theta = (theta_base
                     + 0.0 * self.param_net.forward(torch.FloatTensor(obs)))
        theta = theta + torch.rand_like(theta) * 1e-6
        return theta

    def get_references(self, info_batch: list) -> list:
        """Construct y_ref for each environment from actor_info.

        Analogous to MPCActor.get_references(), but returns y_ref (output
        reference, shape (n_target,)) instead of x_ref (full-state reference).
        """
        y_ref_batch = []
        for info in info_batch:
            traj_step = info["current_step"]
            x_ref = np.asarray(info["x_ref"], dtype=np.float64).T  # (nx, T+1)

            # Extract the y_indices rows: (n_y, T+1)
            y_ref_full = x_ref[self.y_indices, :]  # (n_y, T+1)
            T_plus1 = y_ref_full.shape[1]
            horizon = self.mpc.horizon

            # Slice a window of length horizon starting at traj_step+1
            start = min(traj_step + 1, T_plus1 - 1)
            end = min(traj_step + 1 + horizon, T_plus1)
            remain = horizon - (end - start)
            slices = [y_ref_full[:, start:end]]
            if remain > 0:
                slices.append(np.tile(y_ref_full[:, -1:], (1, remain)))
            y_ref_2d = np.concatenate(slices, axis=1)  # (n_y, horizon)
            y_ref_batch.append(y_ref_2d.T.reshape(-1))  # (n_target,)

        return y_ref_batch

    def forward(self, obs, act=None, actor_info=None, z_past=None):
        """Compute action distribution for rollout collection.

        Mirrors MPCActor.forward() interface exactly.
        z_past: raw (unnormalized) I/O history from PhiEnv — passed directly to QP.
        """
        # Fetch trainable parameters and trajectory reference
        theta = self.get_theta_param(obs)
        traj_ref = self.get_references(actor_info)

        # MPC solve code expects NumPy arrays, not torch tensors
        obs_np = obs if isinstance(obs, np.ndarray) else obs.numpy()
        z_past_np = z_past if (z_past is None or isinstance(z_past, np.ndarray)) else z_past.numpy()

        # Call select_action_batch() or select_action() depending on batch of observations vs. single observation
        
        # Batch (typically training): 
        if obs.ndim > 1:
            action, info, results_dict, optimal_flag = self.mpc.select_action_batch(
                obs_np,
                theta.detach().numpy(),
                traj_ref,
                actor_info,
                z_past_batch=z_past_np,
            )
        # Single (typically eval): 
        else:
            action, info, results_dict, optimal_flag = self.mpc.select_action(
                obs_np,
                theta.detach().numpy(),
                traj_ref,
                z_past=z_past_np,
            )

        # Wrap the MPC action(s) as a Gaussian PPO policy - sampling happens in step()
        action = torch.FloatTensor(np.array(action))
        optimal_flag = torch.FloatTensor(
            np.array(optimal_flag) if not isinstance(optimal_flag, np.ndarray)
            else optimal_flag.flatten()
        )
        dist = self.dist_fn(action)
        logp_a = None
        if act is not None:
            logp_a = dist.log_prob(act)
        return dist, logp_a, info, results_dict, optimal_flag

    def forward_train(self, obs, act, info):
        """Compute action + KKT sensitivity for the PPO gradient update.

        Mirrors MPCActor.forward_train() interface exactly.
        """
        theta = self.get_theta_param(obs)
        action, nabla_pi_ref, nabla_pi_theta, optimal_flag = (
            self.mpc.select_action_batch_train(
                obs.numpy(),
                theta.detach().numpy(),
                None,   # y_ref is retrieved from info_batch["ref_param"]
                info,
            )
        )
        action_th = action
        action_th.requires_grad_()
        dist = self.dist_fn(action_th)
        logp_a = dist.log_prob(act)
        return action_th, dist, logp_a, nabla_pi_ref, nabla_pi_theta, optimal_flag

    def reset(self, idx=None):
        """Reset the I/O history buffer for environment idx."""
        self.mpc.reset(idx)


class PhiMLPActorCritic(MLPActorCritic):
    """Actor-critic container for Phi-MPC RL.

    Subclasses MLPActorCritic (ppo_mpc_utils.py).
    Overrides __init__, act(), and step() to route z_past through the actor QP
    and the critic. Only reset() is inherited unchanged.
    """
    def __init__(
        self,
        env,
        obs_space,
        act_space,
        gamma: float,
        phi_past_init: np.ndarray,
        phi_future_init: np.ndarray,
        mask_future: np.ndarray,
        hidden_dims=(64, 64),
        exploration_init=-1.0,
        activation="tanh",
        actor_config=None,
        critic_obs_dim=None,
    ):
        assert critic_obs_dim is not None
        
        # This class inherits from MLPActorCritic, but we do not want its constructor
        # as it would create the wrong actor.
        nn.Module.__init__(self)
        obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            raise ValueError("PhiMLPActorCritic only supports continuous action spaces.")

        # In stead manually construct the actor that we want (the Phi based one)
        self.actor = PhiMPCActor(
            env, obs_dim, act_dim, (hidden_dims[0],) * 2, activation, gamma,
            phi_past_init, phi_future_init, mask_future,
            exploration_init, actor_config,
        )

        # Critic: unchanged from Shambhus approach except uses z_past (input-output history of length t_ini)
        self.critic = MLPCritic(critic_obs_dim, list(hidden_dims), activation)

    def act(self, obs, info=None, z_past_raw=None):
        """Deterministic action for eval — passes raw z_past to the actor."""
        with torch.no_grad():
            dist, _, _, _, _ = self.actor(obs, actor_info=info, z_past=z_past_raw)
        return dist.mode().detach().numpy()

    def step(self, obs, info=None, z_past_norm=None, z_past_raw=None):
        """Sample an action for rollout collection.

        z_past_norm: normalized z_past tensor — input to the critic.
        z_past_raw:  raw z_past numpy array from PhiEnv — input to the actor QP.
        """
        
        # Call forward() method on the actor with RAW z_past
        dist, _, soln_info, results_dict, optimal_flag = self.actor(
            obs, actor_info=info, z_past=z_past_raw
        )
        
        # Sample from the returned Gaussian distribution
        a = dist.sample()
        
        # Compute the log-probability of the sampled action under the current policy, 
        # which PPO later needs for the policy-ratio/clipping objective
        logp_a = dist.log_prob(a)
        
        # Compute critics value-estimations of current state of envs
        # NB! Estimates based on input-output history, not full state
        # TODO: overly defensive in the function call, this is always okay. Remove.  
        v = self.critic(z_past_norm if z_past_norm is not None else obs)
        return (
            a.cpu().numpy(),
            v.cpu().numpy(),
            logp_a.cpu().numpy(),
            soln_info,
            results_dict,
            optimal_flag,
        )
