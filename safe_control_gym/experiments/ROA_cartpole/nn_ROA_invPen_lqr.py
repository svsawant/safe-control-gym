import os
import sys
import time
import pickle
from collections import defaultdict
from functools import partial
import torch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# from safe_control_gym.envs.benchmark_env import Task
# from safe_control_gym.experiments.base_experiment import BaseExperiment
# from safe_control_gym.utils.configuration import ConfigFactory
# from safe_control_gym.utils.registration import make

from safe_control_gym.experiments.ROA_cartpole.utilities import *
from lyapnov import LyapunovNN, Lyapunov, QuadraticFunction, GridWorld_pendulum
from utilities import balanced_class_weights, dlqr, \
                      get_discrete_linear_system_matrices, onestep_dynamics

np.set_printoptions(threshold=sys.maxsize) # np print full array

class Options(object):
    def __init__(self, **kwargs):
        super(Options, self).__init__()
        self.__dict__.update(kwargs)

OPTIONS = Options(np_dtype              = np.float32,
                  torch_dtype           = torch.float32,
                  eps                   = 1e-8,                            # numerical tolerance
                  saturate              = True,                            # apply saturation constraints to the control input
                  use_zero_threshold    = True,                            # assume the discretization is infinitely fine (i.e., tau = 0)
                  pre_train             = True,                            # pre-train the neural network to match a given candidate in a supervised approach
                  dpi                   = 150,
                  num_cores             = 4,
                  num_sockets           = 1,
                #   tf_checkpoint_path    = "./tmp/lyapunov_function_learning.ckpt"
                )

# Constants
dt = 0.01   # sampling time
g = 9.81    # gravity

# True system parameters
m = 0.15    # pendulum mass
L = 0.5     # pole length
b = 0.1     # rotational friction

# State and action normalizers
theta_max = np.deg2rad(180)                     # angular position [rad]
omega_max = np.deg2rad(360)                     # angular velocity [rad/s]
# u_max     = g * m * L * np.sin(np.deg2rad(60))  # torque [N.m], control action
u_max = 0.5

state_norm = (theta_max, omega_max)
action_norm = (u_max,)

# Dimensions and domains
state_dim     = 2
action_dim    = 1
state_limits  = np.array([[-1., 1.]] * state_dim)
action_limits = np.array([[-1., 1.]] * action_dim)

# Initialize system class and its linearization
pendulum = InvertedPendulum(m, L, b, dt, [state_norm, action_norm])
A, B = pendulum.linearize()
# print("A\n ", A)
# print("B\n ", B)
# dynamics = pendulum.__call__
dynamics = pendulum.__call__

# # test dynamics with state [0.5, 0] and action [0.5]
# x = torch.tensor([0.5, 0.], dtype=OPTIONS.torch_dtype)
# u = torch.tensor([-0.5], dtype=OPTIONS.torch_dtype)
# x = np.array([0.5, 0.])
# u = np.array([-0.5])
# print("x: ", x)
# print("u: ", u)
# state_action = np.concatenate([x, u])
# state_action = torch.cat([x, u])
# print("state_action: ", state_action)

# x_next = dynamics(state_action)
# print("x_next: ", x_next)
# assert x_next.shape == x.shape
# print("Dynamics shape test passed!")

state_constraints = np.array([[-theta_max, theta_max], [-omega_max, omega_max]])

print('state_constraints: ', state_constraints)
num_states = 100

grid_limits = np.array([[-1., 1.], ] * state_dim)
# state_discretization = gridding(state_dim, state_constraints=None, num_states = 100)
state_discretization = GridWorld_pendulum(grid_limits, num_states)
# state_discretization = gridding(state_dim, state_constraints, num_states = 100)
print('state_discretization.all_points.shape: ', state_discretization.all_points.shape)

# Discretization constant
if OPTIONS.use_zero_threshold:
    tau = 0.0
else:
    tau = np.sum(state_discretization.unit_maxes) / 2

print('Grid size: {}'.format(state_discretization.nindex))
print('Discretization constant (tau): {}'.format(tau))

# Set initial safe set as a ball around the origin (in normalized coordinates)
cutoff_radius    = 0.1
initial_safe_set = np.linalg.norm(state_discretization.all_points, ord=2, axis=1) <= cutoff_radius
print('state_discretization.all_points.shape: ', state_discretization.all_points.shape)
print('initial_safe_set.sum(): ', initial_safe_set.shape)

Q = np.identity(state_dim).astype(OPTIONS.np_dtype)     # state cost matrix
Q = np.diag([5, 1])
R = 1* np.identity(action_dim).astype(OPTIONS.np_dtype)    # action cost matrix
# K, P_lqr = safe_learning.utilities.dlqr(A, B, Q, R)
K, P_lqr = dlqr(A, B, Q, R) 
print('K',K)

policy = lambda x: -K @ x
if OPTIONS.saturate:
    # policy = lambda x: np.clip(-K @ x, -u_max, u_max)
    policy = lambda x: np.clip(-K @ x, -1, 1)

cl_dynamics = lambda x: dynamics(np.concatenate([x, policy(x)]))

L_pol = lambda x: np.linalg.norm(-K, 1)

L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x)

lyapunov_function = QuadraticFunction(P_lqr)

# Approximate local Lipschitz constants with gradients
# grad_lyapunov_function = lambda x: 2 * P_lqr @ x
grad_lyapunov_function = lambda x: 2 * torch.tensor(P_lqr, dtype=torch.float32) @ x
# L_v = lambda x: tf.norm(grad_lyapunov_function(x), ord=1, axis=1, keepdims=True)
L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=-1, keepdim=True)

# Initialize Lyapunov class
lyapunov_lqr = Lyapunov(state_discretization, lyapunov_function, cl_dynamics, L_dyn, L_v, tau, policy, initial_safe_set)
# print the first 10 states in the safe set
lyapunov_lqr.discretization.all_points[1:10]
# print('lyaupunov_lqr.discretization.all_points[1:10]\n', lyapunov_lqr.discretization.all_points[1:10])
lyapunov_lqr.update_values()
lyapunov_lqr.update_safe_set()
# print('lyapunov_lqr.safe_set\n', lyapunov_lqr.safe_set[1:10])
# print('lyapunov_lqr.values\n', lyapunov_lqr.values[1:10])
# print('lyapunov_lqr.c_max\n', lyapunov_lqr.c_max)
# print('lyapunov_lqr.safe_set.sum()\n', lyapunov_lqr.safe_set.sum())


horizon = 500
tol = 0.1
# roa, trajectories = compute_roa_pendulum(lyapunov_lqr.discretization, cl_dynamics, horizon, tol, no_traj=False)
compute_new_roa = False
# compute_new_roa = True
roa_file_name = 'roa_pendulum.npy'
traj_file_name = 'traj_pendulum.npy'
if not compute_new_roa:
    # load the pre-saved ROA to avoid re-computation
    roa = np.load(roa_file_name)
    trajectories = np.load(traj_file_name)
else:
    roa, trajectories = compute_roa_pendulum(lyapunov_lqr.discretization, cl_dynamics, horizon, tol, no_traj=False)
    np.save(roa_file_name, roa)
    np.save(traj_file_name, trajectories)
    # exit()

print('True ROA size:{}\n'.format(int(roa.sum())))
print('')


grid              = lyapunov_lqr.discretization
################################ plotting ################################
fig = plt.figure(figsize=(8, 3), dpi=OPTIONS.dpi, frameon=False)
fig.subplots_adjust(wspace=0.35)
plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max, omega_max])))

# ax = plt.subplot(121)
ax = plt.subplot(111)
alpha = 1
colors = [None] * 4
colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
colors[1] = (230/255, 159/255, 0)       # NN  - orange
colors[2] = (0, 114/255, 178/255)       # LQR - blue
colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow

# True ROA
z = roa.reshape(grid.num_points)
print(z.shape)
print(z)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)

# LQR
z = lyapunov_lqr.safe_set.reshape(grid.num_points)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[2],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[2]), alpha=alpha)



# Plot some trajectories
N_traj = 11
skip = int(grid.num_points[0] / N_traj)
sub_idx = np.arange(grid.nindex).reshape(grid.num_points)
sub_idx = sub_idx[::skip, ::skip].ravel()
sub_trajectories = trajectories[sub_idx, :, :]
sub_states = grid.all_points[sub_idx]
for n in range(sub_trajectories.shape[0]):
    x = sub_trajectories[n, 0, :] * np.rad2deg(theta_max)
    y = sub_trajectories[n, 1, :] * np.rad2deg(omega_max)
    ax.plot(x, y, 'k--', linewidth=0.25)
sub_states = grid.all_points[sub_idx]
dx_dt = np.zeros_like(sub_states)
for state_idx in range(sub_states.shape[0]):
    dx_dt[state_idx, :] = (cl_dynamics(sub_states[state_idx, :]) - sub_states[state_idx, :])/dt

# dx_dt = (tf_future_states.eval({tf_states: sub_states}) - sub_states) / dt
dx_dt = dx_dt / np.linalg.norm(dx_dt, ord=2, axis=1, keepdims=True)
ax.quiver(sub_states[:, 0] * np.rad2deg(theta_max), sub_states[:, 1] * np.rad2deg(omega_max), dx_dt[:, 0], dx_dt[:, 1], 
          scale=None, pivot='mid', headwidth=3, headlength=6, color='k')

ax.set_title('ROA of pendulum under an LQR prior policy (l={:.2f})'.format(L))
ax.set_aspect(theta_max / omega_max / 1.2)
ax.set_xlim(plot_limits[0])
ax.set_ylim(plot_limits[1])
ax.set_xlabel(r'angle [deg]')
ax.set_ylabel(r'angular velocity [deg/s]')
ax.xaxis.set_ticks(np.arange(-180, 181, 60))
ax.yaxis.set_ticks(np.arange(-360, 361, 120))

proxy = [plt.Rectangle((0,0), 1, 1, fc=c) for c in colors]    
legend = ax.legend(proxy, [r'Brute-forced ROA'], loc='upper right')
legend.get_frame().set_alpha(1.)



plt.show()