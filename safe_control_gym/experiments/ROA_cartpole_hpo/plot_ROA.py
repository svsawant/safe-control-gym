import numpy as np
import matplotlib.pyplot as plt
import torch
from safe_control_gym.experiments.ROA_cartpole.utilities import *
from lyapnov import GridWorld_pendulum

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
#################################### Constants ####################################
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

############################### System dynamics ################################
# Initialize system class and its linearization
pendulum = InvertedPendulum(m, L, b, dt, [state_norm, action_norm])
A, B = pendulum.linearize()
# print("A\n ", A)
# print("B\n ", B)
# dynamics = pendulum.__call__
dynamics = pendulum.__call__

############################### Discretization ################################
state_constraints = np.array([[-theta_max, theta_max], [-omega_max, omega_max]])
# print('state_constraints: ', state_constraints)
num_states = 100

grid_limits = np.array([[-1., 1.], ] * state_dim)
# state_discretization = gridding(state_dim, state_constraints=None, num_states = 100)
state_discretization = GridWorld_pendulum(grid_limits, num_states)
# state_discretization = gridding(state_dim, state_constraints, num_states = 100)
# print('state_discretization.all_points.shape: ', state_discretization.all_points.shape)

# Discretization constant
if OPTIONS.use_zero_threshold:
    tau = 0.0
else:
    tau = np.sum(state_discretization.unit_maxes) / 2

print('Grid size: {}'.format(state_discretization.nindex))
print('Discretization constant (tau): {}'.format(tau))

################################ plotting ################################
fig = plt.figure(figsize=(8, 3), dpi=OPTIONS.dpi, frameon=False)
fig.subplots_adjust(wspace=0.35)
plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max, omega_max])))

# ax = plt.subplot(121)
ax = plt.subplot(111)
alpha = 1
colors = [None] * 4
colors[0] = (0, 0.9, 1)       # ROA - bluish-green
colors[1] = (1, 0.5, 0)       # NN  - orange
colors[2] = (0, 0, 1)       # LQR - blue
colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow


nn_ROA_data_file =  'nn_roa_pendulum_0.5.npy'
z = np.load(nn_ROA_data_file).reshape(state_discretization.num_points)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[2],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[2]), alpha=alpha)

nn_ROA_data_file =  'nn_roa_pendulum_0.8.npy'
z = np.load(nn_ROA_data_file).reshape(state_discretization.num_points)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)


nn_ROA_data_file =  'nn_roa_pendulum_0.2.npy'
z = np.load(nn_ROA_data_file).reshape(state_discretization.num_points)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[1],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[1]), alpha=alpha)

ax.set_title('ROAs of pendulum under an LQR prior policy')
ax.set_aspect(theta_max / omega_max / 1.2)
ax.set_xlim(plot_limits[0])
ax.set_ylim(plot_limits[1])
ax.set_xlabel(r'angle [deg]')
ax.set_ylabel(r'angular velocity [deg/s]')
ax.xaxis.set_ticks(np.arange(-180, 181, 60))
ax.yaxis.set_ticks(np.arange(-360, 361, 120))

proxy = [plt.Rectangle((0,0), 1, 1, fc=c) for c in colors]    
legend = ax.legend(proxy, [ r'l = 0.8 (better)', r'l = 0.2 (worse)', r'l = 0.5 (true)' ], loc='upper right')
legend.get_frame().set_alpha(1.)



plt.show()