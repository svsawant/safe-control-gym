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

from utilities import *
import time

from pprint import pprint

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
    print('random env\n', random_env.__dir__())
    # print('random env.CTRL_STEPS', random_env.CTRL_STEPS)
    # print('random env.EPISODE_LEN_SEC', random_env.EPISODE_LEN_SEC)
    # print('\n')
    # print('random env.constraints.constraints\n')
    # print(random_env.constraints.state_constraints[0].lower_bounds)
    # print(random_env.constraints.__dir__())
    # print('\n')
    # print effective (true) parmeters
    print('random env\n', random_env.__dir__())
    print('random env.POLE_MASS', random_env.POLE_MASS)
    print('random env.EFFECTIVE_POLE_LENGTH', random_env.EFFECTIVE_POLE_LENGTH)
    print('random env.CART_MASS', random_env.CART_MASS)
    

    # exit()
    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    # print('ctrl\n', ctrl.__dir__())
    # print('ctrl.model\n', ctrl.model.__dir__())
    print('ctrl.model.pole_mass', ctrl.model.pole_mass) # m
    print('ctrl.model.pole_length', ctrl.model.pole_length) # l
    print('ctrl.model.cart_mass', ctrl.model.cart_mass) # M
    m = ctrl.model.pole_mass
    l = ctrl.model.pole_length
    M = ctrl.model.cart_mass

    # exit()
    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes
    
    # state constraints
    state_constraints = np.vstack((random_env.constraints.state_constraints[0].lower_bounds,
                                   random_env.constraints.state_constraints[0].upper_bounds)).T
    # print('state constraints', state_constraints)
    dim_state = ctrl.model.x_sym.shape[0] # state dimension
    
    # lower bound has the shape [-1, -1, -1, -1]
    # override state constraints with self-defined constraints
    state_constraints = np.array([1, 3, 3, 3])
    state_constraints = np.vstack((-1 * state_constraints, \
                                        state_constraints)).T
    num_states = 51
    prec = []
    prec.append(2) # dummy dimension: cartpole is position-invariant
    for _ in range(dim_state - 1):
        prec.append(num_states)

    grids = gridding(dim_state, state_constraints, prec)
    # print('grids.all_points\n', grids.all_points)
    # print('grids.all_points.shape\n', grids.all_points.shape)
    # exit()
    
    # Run the experiment.
    # forward simulation all trajtories from all points in grids
    # compute_new_ROA = False
    compute_new_ROA = True
    if compute_new_ROA:
        before_roa_time = time.time()
        roa = compute_roa_par(grids, env_func, ctrl, no_traj=True)
        after_roa_time = time.time()
        print('time for compute_roa', after_roa_time - before_roa_time)
        # save roa to file
        roa_file_name = 'roa_M_'+repr(M)+\
                           '_m_'+repr(m)+\
                           '_l_'+repr(l)+\
                           '.npy'
        np.save(roa_file_name, roa)
    else:
        # load roa from file
        roa_file_name = 'roa_M_'+repr(M)+\
                           '_m_'+repr(m)+\
                           '_l_'+repr(l)+\
                           '.npy'
        roa = np.load(roa_file_name)

    safe_fraction = np.sum(roa) / len(grids.all_points)
    print('safe fraction', safe_fraction)
    
    # print('roa\n', roa)
    # print('grids.all_points\n', grids.all_points)
    # concatenate all points in grids with roa
    res = np.hstack(( roa.reshape(-1, 1), grids.all_points))
    print('res\n', res)
    ctrl.close()
    random_env.close()
    exit()

    exit()

    for _ in range(n_episodes):
        # convert state point to 
        # Get initial state and create environments
        init_state = grids.all_points[0]
        init_state_dict = {'init_x': init_state[0], 'init_x_dot': init_state[1], \
                           'init_theta': init_state[2], 'init_theta_dot': init_state[3]}
        # init with states in grids
        init_state, _ = random_env.reset(init_state=init_state_dict)
        print('init state in script', init_state)
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
        # experiment.launch_training() # no training here

        if n_steps is None:
            # if n_steps is None, run # episode experiments
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
            # print('trajs_data', trajs_data)
            print('goal reached', trajs_data['info'][-1][1]['goal_reached'])
            exit()
        else:
            # if n_steps is not None, run # step experiments
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        
        post_analysis(trajs_data['obs'][0], trajs_data['action'][0], ctrl.env)

        # Close environments
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    # if save_data:
    #     results = {'trajs_data': all_trajs, 'metrics': metrics}
    #     path_dir = os.path.dirname('./temp-data/')
    #     os.makedirs(path_dir, exist_ok=True)
    #     with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
    #         pickle.dump(results, file)

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
