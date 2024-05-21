import numpy as np
import matplotlib.pyplot as plt
import os


from safe_control_gym.lyapunov.utilities import *

# make the plot resolution higher and make font size bigger
resolution = 100
plt.rcParams['figure.dpi'] = resolution
# plt.rcParams.update({'font.size': 20})


dt = 0.06666666666666667
# state constraints
dim_grid = 4
grid_constraints = np.array([0.5, 0.5, 0.5, 0.5])
# grid_constraints = np.array([0.5, 1.5, 1.57, 1.57])
# grid_constraints = np.array([1, 0.3, 0.2])
grid_constraints = np.vstack((-1 * grid_constraints, \
                                    grid_constraints)).T
# print('state constraints', state_constraints)
# prec = [3, 41, 51, 41]
prec = [2, 2, 2, 2]
grids = gridding(dim_grid, grid_constraints, prec)
num_subplots = np.prod(prec)
colors = ['b', 'g', 'r']
line_styles = ['-', ':', '-.']

# get current script directory
current_dir = os.path.dirname(os.path.realpath(__file__))
solver_name = ['ipopt', 'ipopt_warm_start_with_qrsqp', 'qrsqp']

traj_data_name = 'trajectories_' + solver_name[0] + '.npy'
abs_traj_data_name = os.path.join(current_dir, traj_data_name)
trajs = np.load(abs_traj_data_name, allow_pickle=True)
fig, axs = plt.subplots(5, num_subplots, figsize=(20, 20))
for i in range(num_subplots):
    init_state = grids.all_points[i]
    print('init_state', init_state)
    # exit()
    # state
    max_iter = trajs[i]['state_traj'].shape[0]
    state_time_axis = np.linspace(0, max_iter * dt, max_iter)
    axs[0, i].plot(state_time_axis, trajs[i]['state_traj'][:, 0] ,\
                    color=colors[0], label=solver_name[0], linestyle=line_styles[0])
    axs[0, i].set(ylabel='x [m]')            
    axs[0, i].set_title(f'init_state: {init_state}') # set title
    axs[1, i].plot(state_time_axis, trajs[i]['state_traj'][:, 1], \
                   color=colors[0],  label=solver_name[0], linestyle=line_styles[0])
    axs[1, i].set(ylabel='x_dot [m/s]')
    axs[2, i].plot(state_time_axis, trajs[i]['state_traj'][:, 2], \
                   color=colors[0], label=solver_name[0], linestyle=line_styles[0])
    axs[2, i].set(ylabel='theta [rad]')
    axs[3, i].plot(state_time_axis, trajs[i]['state_traj'][:, 3], \
                   color=colors[0], label=solver_name[0], linestyle=line_styles[0])
    axs[3, i].set(ylabel='theta_dot [rad/s]')
    # input
    input_time_axis = np.linspace(0, max_iter * dt, max_iter - 1)
    axs[4, i].plot(input_time_axis, trajs[i]['input_traj'][:, 0], \
                   color=colors[0], label=solver_name[0], linestyle=line_styles[0])
    axs[4, i].set(xlabel='time [s]', ylabel='force [N]')

traj_ipopt = trajs

traj_data_name = 'trajectories_' + solver_name[1] + '.npy'
abs_traj_data_name = os.path.join(current_dir, traj_data_name)
trajs = np.load(abs_traj_data_name, allow_pickle=True)
# fig, axs = plt.subplots(5, num_subplots, figsize=(20, 20))
for i in range(num_subplots):
    init_state = grids.all_points[i]
    # exit()
    # state
    max_iter = trajs[i]['state_traj'].shape[0]
    state_time_axis = np.linspace(0, max_iter * dt, max_iter)
    axs[0, i].plot(state_time_axis, trajs[i]['state_traj'][:, 0] , \
                   color=colors[1], label=solver_name[1], linestyle=line_styles[1], marker = 'o')
    axs[0, i].set(ylabel='x [m]')            
    axs[0, i].set_title(f'init_state: {init_state}') # set title
    axs[1, i].plot(state_time_axis, trajs[i]['state_traj'][:, 1], \
                   color=colors[1], label=solver_name[1], linestyle=line_styles[1], marker = 'o')
    axs[1, i].set(ylabel='x_dot [m/s]')
    axs[2, i].plot(state_time_axis, trajs[i]['state_traj'][:, 2],  \
                   color=colors[1], label=solver_name[1], linestyle=line_styles[1], marker = 'o')
    axs[2, i].set(ylabel='theta [rad]')
    axs[3, i].plot(state_time_axis, trajs[i]['state_traj'][:, 3],  \
                   color=colors[1], label=solver_name[1], linestyle=line_styles[1], marker = 'o')
    axs[3, i].set(ylabel='theta_dot [rad/s]')
    # input
    input_time_axis = np.linspace(0, max_iter * dt, max_iter - 1)
    axs[4, i].plot(input_time_axis, trajs[i]['input_traj'][:, 0], \
                   color=colors[1], label=solver_name[1], linestyle=line_styles[1], marker = 'o')
    axs[4, i].set(xlabel='time [s]', ylabel='force [N]')


traj_warmp_start = trajs

traj_data_name = 'trajectories_' + solver_name[2] + '.npy'
abs_traj_data_name = os.path.join(current_dir, traj_data_name)
trajs = np.load(abs_traj_data_name, allow_pickle=True)
# fig, axs = plt.subplots(5, num_subplots, figsize=(20, 20))
for i in range(num_subplots):
    init_state = grids.all_points[i]
    # exit()
    # state
    max_iter = trajs[i]['state_traj'].shape[0]
    state_time_axis = np.linspace(0, max_iter * dt, max_iter)
    axs[0, i].plot(state_time_axis, trajs[i]['state_traj'][:, 0] , \
                   color=colors[2], label=solver_name[2], linestyle=line_styles[2], marker = 'x')
    axs[0, i].set(ylabel='x [m]')            
    axs[0, i].set_title(f'init_state: {init_state}') # set title
    axs[1, i].plot(state_time_axis, trajs[i]['state_traj'][:, 1], \
                   color=colors[2], label=solver_name[2], linestyle=line_styles[2], marker = 'x')
    axs[1, i].set(ylabel='x_dot [m/s]')
    axs[2, i].plot(state_time_axis, trajs[i]['state_traj'][:, 2], \
                   color=colors[2], label=solver_name[2], linestyle=line_styles[2], marker = 'x')
    axs[2, i].set(ylabel='theta [rad]')
    axs[3, i].plot(state_time_axis, trajs[i]['state_traj'][:, 3], \
                   color=colors[2], label=solver_name[2], linestyle=line_styles[2], marker = 'x')
    axs[3, i].set(ylabel='theta_dot [rad/s]')
    # input
    input_time_axis = np.linspace(0, max_iter * dt, max_iter - 1)
    axs[4, i].plot(input_time_axis, trajs[i]['input_traj'][:, 0], \
                   color=colors[2], label=solver_name[2], linestyle=line_styles[2], marker = 'x')
    axs[4, i].set(xlabel='time [s]', ylabel='force [N]')

# make the legend font size bigger


plt.legend(fontsize=30)


############################### RMSE ########################################
# compute the rmse between IPOPT and warm-starting for each state and input at each time step
rmse = np.zeros((5, num_subplots))
for i in range(num_subplots):
    rmse[0, i] = np.sqrt(np.mean(np.square(traj_ipopt[i]['state_traj'][:, 0] - traj_warmp_start[i]['state_traj'][:, 0])))
    rmse[1, i] = np.sqrt(np.mean(np.square(traj_ipopt[i]['state_traj'][:, 1] - traj_warmp_start[i]['state_traj'][:, 1])))
    rmse[2, i] = np.sqrt(np.mean(np.square(traj_ipopt[i]['state_traj'][:, 2] - traj_warmp_start[i]['state_traj'][:, 2])))
    rmse[3, i] = np.sqrt(np.mean(np.square(traj_ipopt[i]['state_traj'][:, 3] - traj_warmp_start[i]['state_traj'][:, 3])))
    rmse[4, i] = np.sqrt(np.mean(np.square(traj_ipopt[i]['input_traj'][:, 0] - traj_warmp_start[i]['input_traj'][:, 0])))

print('rmse.shape', rmse.shape)

# plot the rmse (y axis is the rmse, x axis is time)
fig, axs = plt.subplots(1, 5, figsize=(20, 5))
time_axis = np.arange(0, num_subplots)
axs[0].plot(time_axis, rmse[0, :], label='x')
axs[0].set_title('x')
axs[0].set(xlabel='time [s]', ylabel='rmse')

axs[1].plot(time_axis, rmse[1, :], label='x_dot')
axs[1].set_title('x_dot')
axs[1].set(xlabel='time [s]', ylabel='rmse')

axs[2].plot(time_axis, rmse[2, :], label='theta')
axs[2].set_title('theta')
axs[2].set(xlabel='time [s]', ylabel='rmse')
axs[3].plot(time_axis, rmse[3, :], label='theta_dot')
axs[3].set_title('theta_dot')
axs[3].set(xlabel='time [s]', ylabel='rmse')
axs[4].plot(time_axis, rmse[4, :], label='force')
axs[4].set_title('force')
axs[4].set(xlabel='time [s]', ylabel='rmse')
# add overall title
fig.suptitle('rmse between IPOPT and warm-starting + SQP')

plt.show()