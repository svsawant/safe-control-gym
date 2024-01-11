import os
import sys
import time
import pickle
from collections import defaultdict
from functools import partial
import torch
from torchviz import make_dot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.experiments.ROA_cartpole.utilities import *
from lyapnov import LyapunovNN, Lyapunov, QuadraticFunction, GridWorld_pendulum
from utilities import balanced_class_weights, dlqr, \
                      get_discrete_linear_system_matrices, onestep_dynamics

# set random seed for reproducibility
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

np.set_printoptions(threshold=sys.maxsize) # np print full array
# torch.autograd.set_detect_anomaly(True)

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

# detect torch device
# myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myDevice = torch.device("cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


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

# Set initial safe set as a ball around the origin (in normalized coordinates)
cutoff_radius    = 0.1
initial_safe_set = np.linalg.norm(state_discretization.all_points, ord=2, axis=1) <= cutoff_radius
# print('state_discretization.all_points.shape: ', state_discretization.all_points.shape)
# print('initial_safe_set.sum(): ', initial_safe_set.shape)

########################## define LQR policy ##############################
Q = np.identity(state_dim).astype(OPTIONS.np_dtype)     # state cost matrix
Q = np.diag([5, 1])
R = 1* np.identity(action_dim).astype(OPTIONS.np_dtype)    # action cost matrix
K, P_lqr = dlqr(A, B, Q, R) 

policy = lambda x: -K @ x
if OPTIONS.saturate:
    policy = lambda x: np.clip(-K @ x, -1, 1)

###############  closed-loop dynamics and Lipschitz constants ##############
    
cl_dynamics = lambda x: dynamics(np.concatenate([x, policy(x)]))
L_pol = lambda x: np.linalg.norm(-K, 1)
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x)

########################## define Lyapunov LQR ##########################
lyapunov_function = QuadraticFunction(P_lqr)
# Approximate local Lipschitz constants with gradients
grad_lyapunov_function = lambda x: 2 * torch.tensor(P_lqr, dtype=torch.float32) @ x
L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=-1, keepdim=True)
# Initialize Lyapunov class
lyapunov_lqr = Lyapunov(state_discretization, lyapunov_function, cl_dynamics, L_dyn, L_v, tau, policy, initial_safe_set)
lyapunov_lqr.update_values()
lyapunov_lqr.update_safe_set()
# print('lyapunov_lqr.c_max\n', lyapunov_lqr.c_max)
print('lyapunov_lqr.safe_set.sum()\n', lyapunov_lqr.safe_set.sum())

########################## compute ROA ################################
horizon = 500
tol = 0.1
# compute_new_roa = False
compute_new_roa = True
script_dir = os.path.dirname(__file__)
roa_file_name = 'roa_pendulum.npy'
traj_file_name = 'traj_pendulum.npy'
# append the file name to the current path
roa_file_name = os.path.join(script_dir, roa_file_name)
traj_file_name = os.path.join(script_dir, traj_file_name)
if not compute_new_roa:
    # load the pre-saved ROA to avoid re-computation
    roa = np.load(roa_file_name)
    trajectories = np.load(traj_file_name)
else:
    brute_force_start_time = time.time()
    roa, trajectories = compute_roa_pendulum(lyapunov_lqr.discretization, cl_dynamics, horizon, tol, no_traj=False)
    brute_force_end_time = time.time()
    np.save(roa_file_name, roa)
    np.save(traj_file_name, trajectories)
    # exit()

print('True ROA size:{}\n'.format(int(roa.sum())))
print('')
######################## define Lyapunov NN ########################
# initialize Lyapunov NN
layer_dim = [64, 64, 64]
# layer_dim = [128, 128, 128]
activations = [torch.nn.Tanh(), torch.nn.Tanh(), torch.nn.Tanh()]
nn = LyapunovNN(state_dim, layer_dim, activations)
# test_state = np.array([0.0, 0.0])
# print('nn(test_state)', nn(test_state))
test_state = np.array([0.5, 0.5]) # dummy test input
test_output = nn(test_state)
print('test_output', test_output) # dummy test output
make_dot(test_output, params=dict(nn.named_parameters())).render("attached", format="png")
# # nn.print_manual_kernel()
# exit()
print('nn\n', nn)
for name, param in nn.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

# approximate local Lipschitz constant with gradient
grad_lyapunov_function = \
    lambda x: torch.autograd.grad(nn(x), x, \
                    torch.ones_like(nn(x)), allow_unused=True,)[0]
lyapunov_nn = Lyapunov(state_discretization, nn, \
                          cl_dynamics, L_dyn, L_v, tau, policy, \
                          initial_safe_set)
lyapunov_nn.update_values()
lyapunov_nn.update_safe_set()

#########################################################################
# train the parameteric Lyapunov candidate in order to expand the verifiable
# safe set toward the brute-force safe set
test_classfier_loss = []
test_decrease_loss   = []
roa_estimate         = np.copy(lyapunov_nn.safe_set)

# grid              = lyapunov_lqr.discretization
grid              = lyapunov_nn.discretization
c_max             = [lyapunov_nn.c_max, ]
safe_set_fraction = [lyapunov_nn.safe_set.sum() / grid.nindex, ]
print('safe_set_fraction', safe_set_fraction)
######################### traning hyperparameters #######################
outer_iters = 20
inner_iters = 10
horizon     = 100
test_size   = int(1e4)

safe_level = 1
# lagrange_multiplier = 1000
lagrange_multiplier = 5000
level_multiplier = 1.3
learning_rate = 5e-3
# learning_rate = 1e-1
batch_size    = int(1e3)

# optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
# print('optimizer\n', optimizer)
for name, param in lyapunov_nn.lyapunov_function.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
# print('lyaunov_nn.lyapunov_function.input_dim', lyapunov_nn.lyapunov_function.input_dim)
# print('lyaunov_nn.lyapunov_function.num_layers', lyapunov_nn.lyapunov_function.num_layers)
# print('lyaunov_nn.lyapunov_function.kernel', lyapunov_nn.lyapunov_function.kernel)
# exit()
############################# training loop #############################
training_start_time = time.time()
print('Current metrics ...')
c = lyapunov_nn.c_max
num_safe = lyapunov_nn.safe_set.sum()
print('Safe level (c_k): {}'.format(c))
print('Safe set size: {} ({:.2f}% of grid, \
        {:.2f}% of ROA)\n'.format(int(num_safe), \
        100 * num_safe / grid.nindex, 100 * num_safe / roa.sum()))
print('')
time.sleep(0.5)
for _ in range(outer_iters):
    print('Iteration (k): {}'.format(len(c_max)))
    # time.sleep(0.5)

    ## Identify the "gap" states, i.e., those between V(c_k) 
    ## and V(a * c_k) for a > 1
    c = lyapunov_nn.c_max
    idx_small = lyapunov_nn.values.ravel() <= c
    idx_big   = lyapunov_nn.values.ravel() <= level_multiplier * c
    idx_gap   = np.logical_and(idx_big, ~idx_small)

    ## Forward-simulate "gap" states to determine 
    ## which ones we can add to our ROA estimate
    gap_states = grid.all_points[idx_gap]
    gap_future_values = np.zeros((gap_states.shape[0], 1))
    for state_idx in range(gap_states.shape[0]):
        # !! when using dynamics, the state can go out of the bound
        for _ in range(horizon):
            gap_states[state_idx] = np.reshape(cl_dynamics(gap_states[state_idx]), -1)
        gap_future_values[state_idx] = (lyapunov_nn.lyapunov_function(\
                                    gap_states[state_idx]).detach().numpy())
    roa_estimate[idx_gap] |= (gap_future_values <= c).ravel()

    ## Identify the class labels for our current ROA estimate 
    ## and the expanded level set
    target_idx = np.logical_or(idx_big, roa_estimate)
    target_set = grid.all_points[target_idx]
    target_labels = roa_estimate[target_idx]\
                    .astype(OPTIONS.np_dtype).reshape([-1, 1])
    idx_range = target_set.shape[0]

    ## test set
    idx_test = np.random.randint(0, idx_range, size=(test_size, ))
    test_set = target_set[idx_test]
    test_labels = target_labels[idx_test]

    # stochastic gradient descent for classification
    for _ in range(inner_iters):
        lyapunov_nn.lyapunov_function.train()
        # training step
        # safe_level = lyapunov_nn.c_max
        idx_batch_eval = np.random.randint(0, idx_range, size=(batch_size, ))
        # fix the batch from 0 to batch_size
        # idx_batch_eval = np.arange(0, idx_range)
        # print('idx_batch_eval', idx_batch_eval.T)
        training_states = target_set[idx_batch_eval]
        num_training_states = training_states.shape[0]
        
        # True class labels, converted from Boolean ROA labels {0, 1} to {-1, 1}
        roa_labels = target_labels[idx_batch_eval]
        class_label = 2 * roa_labels - 1
        class_label = torch.tensor(class_label, dtype=torch.float32, device=myDevice, requires_grad=False)
        # print('class_label', class_label.T)
        # Signed, possibly normalized distance from the decision boundary
        decision_distance_for_states = torch.zeros((num_training_states, 1), dtype=torch.float32, device=myDevice)                                                   
        for state_idx in range(num_training_states):
            decision_distance_for_states[state_idx] = lyapunov_nn.lyapunov_function(training_states[state_idx])
        decision_distance = safe_level - decision_distance_for_states

        # Perceptron loss with class weights (here all classes are weighted equally)
        class_weights, class_counts = balanced_class_weights(roa_labels.astype(bool))
        # convert class_weights to torch tensor
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=myDevice, requires_grad=False)
        classifier_loss = class_weights * torch.max(- class_label * decision_distance, torch.zeros_like(decision_distance, device=myDevice)) 
        # classifier_loss =  torch.max(- class_label * decision_distance, torch.zeros_like(decision_distance, device=myDevice)) 
        # print('classifier_loss', classifier_loss.T)
        # Enforce decrease constraint with Lagrangian relaxation
        torch_dv_nn = torch.zeros((num_training_states, 1), dtype=torch.float32, device=myDevice, requires_grad=False)
        for state_idx in range(num_training_states):
            future_state = np.reshape(cl_dynamics(training_states[state_idx]), -1)
            torch_dv_nn[state_idx] = lyapunov_nn.lyapunov_function(future_state) - \
                                     lyapunov_nn.lyapunov_function(training_states[state_idx])

        roa_labels = torch.tensor(roa_labels, dtype=torch.float32, device=myDevice, requires_grad=False)
        training_states_forwards = torch.zeros((num_training_states, 1), dtype=torch.float32, device=myDevice, requires_grad=False)
        for state_idx in range(num_training_states):
            training_states_forwards[state_idx] = lyapunov_nn.lyapunov_function(training_states[state_idx])
        
        decrease_loss = roa_labels * torch.max(torch_dv_nn, torch.zeros_like(torch_dv_nn, device=myDevice))  \
                            /(training_states_forwards + OPTIONS.eps)
        # print('decrease_loss', decrease_loss.T)
        loss = torch.mean(classifier_loss + lagrange_multiplier * decrease_loss)
        # make_dot(classifier_loss)
        print('loss', loss)
        optimizer.zero_grad() # zero gradiants for every batch !!
        loss.backward()
        # loss.backward(retain_graph=True)
        # print('nn.lyapunov_function.layers[0].weight\n', lyapunov_nn.lyapunov_function.layers[0].weight)
        # # print('nn.lyapunov_function.layers[1].weight\n', lyapunov_nn.lyapunov_function.layers[1].weight)
        # # print('nn.lyapunov_function.layers[2].weight\n', lyapunov_nn.lyapunov_function.layers[2].weight)
        # # print('nn.lyapunov_function.layers[3].weight\n', lyapunov_nn.lyapunov_function.layers[3].weight)
        # print('nn.lyapunov_function.linear1.weight\n', lyapunov_nn.lyapunov_function.linear1.weight)

        # print('nn.lyapunov_function.layers[0].weight.grad\n', lyapunov_nn.lyapunov_function.layers[0].weight.grad)
        # # print('nn.lyapunov_function.layers[1].weight.grad\n', lyapunov_nn.lyapunov_function.layers[1].weight.grad)
        # # print('nn.lyapunov_function.layers[2].weight.grad\n', lyapunov_nn.lyapunov_function.layers[2].weight.grad)
        # # print('nn.lyapunov_function.layers[3].weight.grad\n', lyapunov_nn.lyapunov_function.layers[3].weight.grad)
        # print('nn.lyapunov_function.linear1.weight.grad\n', lyapunov_nn.lyapunov_function.linear1.weight.grad)

        # test_state_0 = np.array([0.0, 0.0])
        # test_state_1 = np.array([0.5, 0.5])
        # test_state_2 = np.array([1.0, 1.0])
        # value_before_0 = lyapunov_nn.lyapunov_function(test_state_0)
        # value_before_1 = lyapunov_nn.lyapunov_function(test_state_1)
        # value_before_2 = lyapunov_nn.lyapunov_function(test_state_2)
        optimizer.step()
        lyapunov_nn.lyapunov_function.update_kernel()
        # lyapunov_nn.lyapunov_function.print_manual_kernel()
        # value_after_0 = lyapunov_nn.lyapunov_function(test_state_0)
        # value_after_1 = lyapunov_nn.lyapunov_function(test_state_1)
        # value_after_2 = lyapunov_nn.lyapunov_function(test_state_2)
        # print('nn.lyapunov_function.layers[0].weight\n', lyapunov_nn.lyapunov_function.layers[0].weight)
        # print('nn.lyapunov_function.linear1.weight\n', lyapunov_nn.lyapunov_function.linear1.weight)
        # print('value_before_0', value_before_0)
        # print('value_after_0', value_after_0)
        # print('value_before_1', value_before_1)
        # print('value_after_1', value_after_1)
        # print('value_before_2', value_before_2)
        # print('value_after_2', value_after_2)
        # input('press enter to continue')

        # record losses 

    
    ## Update Lyapunov values and ROA estimate, 
    ## based on new parameter values
    lyapunov_nn.update_values()  
    lyapunov_nn.update_safe_set()
    roa_estimate |= lyapunov_nn.safe_set

    c_max.append(lyapunov_nn.c_max)
    safe_set_fraction.append(lyapunov_nn.safe_set.sum() / grid.nindex)
    print('Current safe level (c_k): {}'.format(c_max[-1]))
    print('Safe set size: {} ({:.2f}% of grid, {:.2f}% of ROA)\n'.format(
                            int(lyapunov_nn.safe_set.sum()), \
                            100 * safe_set_fraction[-1], \
                            100 * safe_set_fraction[-1] * roa.size / roa.sum()\
                                ))
training_end_time = time.time()
print('c_max', c_max)
print('safe_set_fraction', safe_set_fraction)
print('Training time: {:.2f} s'.format(training_end_time - training_start_time))
if brute_force_end_time is not None:
    print('Brute-force time: {:.2f} s'.format(brute_force_end_time - brute_force_start_time))

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
# print(z.shape)
# print(z)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)

# # Neural network
z = lyapunov_nn.safe_set.reshape(grid.num_points)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[1],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[1]), alpha=alpha)

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
legend = ax.legend(proxy, [r'Brute-forced ROA', r'NN ROA', r'LQR'], loc='upper right')
legend.get_frame().set_alpha(1.)



plt.show()