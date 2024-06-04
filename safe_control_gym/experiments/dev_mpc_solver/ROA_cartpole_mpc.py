'''A LQR and iLQR example.'''

import os
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from safe_control_gym.lyapunov.utilities import *
import time

def run(gui=False, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running LQR and iLQR experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    # print(config)
    # exit()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)
    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    print('ctrl.model.dt', ctrl.model.dt)
    # exit()

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes
    
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
    
    # Run the experiment.
    # forward simulation all trajtories from all points in grids
    time_before = time.time()
    roa, trajs = compute_roa(grids, env_func, ctrl, no_traj=False)
    time_after = time.time()
    print('time to compute roa', time_after - time_before)

    # plot the trajs on a figure with prec[0]*prec[1]*prec[2]*prec[3] subplots
    num_subplots = np.prod(prec)
    fig, axs = plt.subplots(5, num_subplots, figsize=(20, 20))
    for i in range(num_subplots):
        init_state = grids.all_points[i]
        print('init_state', init_state)
        # exit()
        # state
        max_iter = trajs[i]['state_traj'].shape[0]
        state_time_axis = np.linspace(0, max_iter * ctrl.model.dt, max_iter)
        axs[0, i].plot(state_time_axis, trajs[i]['state_traj'][:, 0])
        axs[0, i].set(ylabel='x [m]')            
        axs[0, i].set_title(f'init_state: {init_state}') # set title
        axs[1, i].plot(state_time_axis, trajs[i]['state_traj'][:, 1])
        axs[1, i].set(ylabel='x_dot [m/s]')
        axs[2, i].plot(state_time_axis, trajs[i]['state_traj'][:, 2])
        axs[2, i].set(ylabel='theta [rad]')
        axs[3, i].plot(state_time_axis, trajs[i]['state_traj'][:, 3])
        axs[3, i].set(ylabel='theta_dot [rad/s]')
        # input
        input_time_axis = np.linspace(0, max_iter * ctrl.model.dt, max_iter - 1)
        axs[4, i].plot(input_time_axis, trajs[i]['input_traj'][:, 0])
        axs[4, i].set(xlabel='time [s]', ylabel='force [N]')
    
    # solver_name = ['ipopt', 'sqpmethods', 'qrsqp']
    solver_name = ['ipopt',  'ipopt_warm_start_with_qrsqp', 'qrsqp']
    traj_data_name = 'trajectories_' + solver_name[1] + '.npy'
    np.save(traj_data_name, trajs)

    # concatenate all points in grids with roa
    res = np.hstack(( roa.reshape(-1, 1), grids.all_points))
    print('res\n', res)
    ctrl.close()
    random_env.close()
    plt.show()
    exit()
    

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))



def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine iLQR's success.

    Args:
        state_stack (ndarray): The list of observations of iLQR in the latest run.
        input_stack (ndarray): The list of inputs of iLQR in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx)
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    plt.show()


def wrap2pi_vec(angle_vec):
    '''Wraps a vector of angles between -pi and pi.

    Args:
        angle_vec (ndarray): A vector of angles.
    '''
    for k, angle in enumerate(angle_vec):
        while angle > np.pi:
            angle -= np.pi
        while angle <= -np.pi:
            angle += np.pi
        angle_vec[k] = angle
    return angle_vec


if __name__ == '__main__':
    run()
