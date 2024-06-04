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
from safe_control_gym.lyapunov.lyapunov import GridWorld

import time
import sys

def run(config, gui=False, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running MPC experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''


    # print(config)
    # exit()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    random_env = env_func(gui=False)
    # print('env_func.func.__dir__()', env_func.func.__dir__())
    # print('random_env.__dir__()', random_env.__dir__())
    # print('random_env.INERTIAL_PROP', random_env.INERTIAL_PROP)
    # print('random_env.POLE_MASS', random_env.POLE_MASS)
    # print('random_env.POLE_LENGTH', random_env.EFFECTIVE_POLE_LENGTH)
    # print('random_env.CART_MASS', random_env.CART_MASS)
    # exit()
    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    # print(ctrl.__dir__())
    print('ctrl.model.__dir__()', ctrl.model.__dir__())
    # print('ctrl.model.pole_mass', ctrl.model.pole_mass)
    # print('ctrl.model.pole_length', ctrl.model.pole_length)
    # print('ctrl.model.cart_mass', ctrl.model.cart_mass)
    # print('ctrl.model.__dir__()', ctrl.model.__dir__())
    # print(ctrl.model.x_sym.shape[0])
    ctrl.reset()
    x_guess = np.zeros((ctrl.model.x_sym.shape[0], 1))
    u_guess = np.zeros((ctrl.model.u_sym.shape[0], 1))
    test_delta_x = np.zeros((ctrl.model.x_sym.shape[0], 1))
    test_delta_u = np.zeros((ctrl.model.u_sym.shape[0], 1))
    # Ad = ctrl.linear_dynamics_func(x0=test_delta_x, p=test_delta_u,
    #                                     x_guess=x_guess, u_guess=u_guess)['Ad']
    # Bd = ctrl.linear_dynamics_func(x0=test_delta_x, p=test_delta_u,
    #                                     x_guess=x_guess, u_guess=u_guess)['Bd']
    # print('Ad', Ad)
    # print('Bd', Bd)
    # print('Ad.shape', Ad.shape)
    # print('Ad type', type(Ad))d
    # exit()
    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes
    
    # ctrl.
    grids = gridding(ctrl.model.x_sym.shape[0], )
    
    
    # Run the experiment.
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()

        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
        # experiment.launch_training() # no training here

        time_before = time.time()
        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)
        time_after = time.time()

        print('Time taken:', time_after - time_before)
        print('goal reached', trajs_data['info'][-1][-1]['goal_reached'])
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

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))
    
    
def gridding(state_dim, use_zero_threshold = True):
    # Number of states along each dimension
    num_states = 251

    # State grid
    grid_limits = np.array([[-1., 1.], ] * state_dim)
    state_discretization = GridWorld(grid_limits, num_states)

    # Discretization constant
    if use_zero_threshold:
        tau = 0.0
    else:
        tau = np.sum(state_discretization.unit_maxes) / 2

    print('Grid size: {}'.format(state_discretization.nindex))
    print('Discretization constant (tau): {}'.format(tau))
    return state_discretization

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
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
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
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
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
    # check whether the config yaml file exists
    ctrl_config_file_name = './config_overrides/cartpole/mpc_acados_cartpole_stabilization.yaml'
    env_config_file_name = './config_overrides/cartpole/cartpole_stabilization.yaml'
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    abs_ctrl_config_file_path = os.path.join(current_script_path, ctrl_config_file_name)
    abs_env_config_file_path = os.path.join(current_script_path, env_config_file_name)
    if not os.path.exists(abs_ctrl_config_file_path):
        print(f"Config file {abs_ctrl_config_file_path} missing.")
        sys.exit(1)
    elif not os.path.exists(abs_env_config_file_path):
        print(f"Config file {abs_env_config_file_path} missing.")
        sys.exit(1)
    else:
        print(f"Config file exists.")
    sys.argv[1:] = ['--algo', 'mpc_acados',
                    '--task', 'cartpole',
                    '--overrides',
                    abs_ctrl_config_file_path, abs_env_config_file_path,
                    '--seed', '20',
                    ]
    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    run(config)
