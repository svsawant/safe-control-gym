

'''A LQR and iLQR example.'''

import os
import pickle
from collections import defaultdict
from functools import partial
import torch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from safe_control_gym.experiments.ROA_cartpole.utilities import *
from lyapnov import LyapunovNN, Lyapunov

from pprint import pprint

class Options(object):
    def __init__(self, **kwargs):
        super(Options, self).__init__()
        self.__dict__.update(kwargs)

OPTIONS = Options(
                #   np_dtype              = safe_learning.config.np_dtype,
                #   tf_dtype              = safe_learning.config.dtype,
                  eps                   = 1e-8,                            # numerical tolerance
                  saturate              = True,                            # apply saturation constraints to the control input
                  use_zero_threshold    = True,                            # assume the discretization is infinitely fine (i.e., tau = 0)
                  pre_train             = True,                            # pre-train the neural network to match a given candidate in a supervised approach
                  dpi                   = 150,
                  num_cores             = 4,
                  num_sockets           = 1,
                #   tf_checkpoint_path    = "./tmp/lyapunov_function_learning.ckpt"
                )


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
    print('\n')

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    print('ctrl\n', ctrl.__dir__())
    print('ctrl.model\n', ctrl.model.__dir__())

    # print('ctrl.dfdx\n', ctrl.model.df_func(np.array([[1], [1], [1], [1]]), np.array(1)))
    
    # print('L_dyn\n', L_dyn(np.array([[1], [1], [1], [1]]), np.array(1)))
    
    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes
    
    ############### state discretization and initial safe set #################
    # state constraints
    state_constraints = np.vstack((random_env.constraints.state_constraints[0].lower_bounds,
                                   random_env.constraints.state_constraints[0].upper_bounds)).T
    # print('state constraints', state_constraints)
    state_dim = ctrl.model.x_sym.shape[0] # state dimension
    grids = gridding(state_dim, state_constraints, num_states = 4)
    # print(grids.all_points)
    # print('grid.nindex', grids.nindex)
    # print('grid.ndim', grids.ndim)
    # exit()
    # Discretization constant
    if OPTIONS.use_zero_threshold:
        tau = 0.0
    else:
        tau = np.sum(grids.unit_maxes) / 2

    # Set initial safe set as a ball around the origin (in normalized coordinates)
    # cutoff_radius    =  2.0 
    cutoff_radius    = 0.1
    initial_safe_set = np.linalg.norm(grids.all_points, ord=2, axis=1) <= cutoff_radius
    
    
    ################# closed-loop dynamics Lipschitz constant ################
    # TODO: check if this is correct (by Mingxuan)
    # # dynamics (linear approximation)
    L_dyn = lambda x: np.linalg.norm(ctrl.model.df_func(x, ctrl.select_action(x))[0], 1) + \
                         np.linalg.norm(ctrl.model.df_func(x, ctrl.select_action(x))[1], 1)
    
    # squish the test state to be (4,) instead of (4, 1)
    test_state = np.array([[1], [1], [1], [1]]).reshape(-1)
    print('test_state shape', test_state.shape)
    print('selected action', ctrl.select_action(test_state))
    print('L_dyn', L_dyn(test_state))

    ######################## define Lyapunov NN ########################
    # initialize Lyapunov NN
    layer_dim = [64, 64, 64]
    activations = [torch.nn.Tanh(), torch.nn.Tanh(), torch.nn.Tanh()]
    nn = LyapunovNN(state_dim, layer_dim, activations)
    values = nn.forward(np.array([[1], [1], [1], [1]]))
    print('values', values)
    
    # approximate local Lipschitz constant with gradient
    grad_lyapunov_function = \
        lambda x: torch.autograd.grad(nn.forward(x), x, \
                        torch.ones_like(nn.forward(x)), allow_unused=True)[0]
    
    # test grad_lyapunov_function with a random state
    test_state = np.array([[1], [1], [1], [1]]).reshape(-1)
    # convert test_state to torch tensor
    test_state = torch.tensor(test_state, dtype=torch.float32, requires_grad=True)
    print('test_state shape', test_state.shape)
    res = grad_lyapunov_function(test_state)
    print('grad_lyapunov_function', res)
    print('grad_lyapunov_function shape', res.shape)
    # exit()

    # L_v = lambda x: tf.norm(grad_lyapunov_function(x), ord=1, axis=1, keepdims=True)
    L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=0, keepdim=True)
    dynamics = lambda x, u: ctrl.model.fc_func(x, u)
    policy = lambda x: ctrl.select_action(x)

    # test dynamics with a random state and action
    print('dynamics\n', dynamics(np.array([[1], [1], [1], [1]]).reshape(-1), np.array([[1]])))
    # initialize Lyapunov class
    lyapunov_nn = Lyapunov(grids, nn, \
                          dynamics, L_dyn, L_v, tau, policy, \
                          initial_safe_set)
    lyapunov_nn.update_values()
    # print('lyapunov_nn.values', lyapunov_nn.values)
    lyapunov_nn.update_safe_set()
    print('lyapunov_nn.safe_set\n', lyapunov_nn.safe_set)
    # concatenate all points in grids with roa as a sanity check
    # res = np.hstack(( lyapunov_nn.safe_set.reshape(-1, 1), grids.all_points))
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print('res\n', res)
    #########################################################################
    # train the parameteric LYapunov candidate in order to expand the verifiable
    # safe set toward the brute-force safe set
    print('c_max', lyapunov_nn.c_max)
    ############### initialization cell to restore parameters ###############
    test_classfier_loss = []
    test_decrease_loss   = []
    roa_estimate         = np.copy(lyapunov_nn.safe_set)

    grid              = lyapunov_nn.discretization
    c_max             = [lyapunov_nn.c_max, ]
    safe_set_fraction = [lyapunov_nn.safe_set.sum() / grid.nindex, ]
    print('safe_set_fraction', safe_set_fraction)
    ######################### traning hyperparameters #######################
    outer_iters = 20
    inner_iters = 10
    horizon     = 100
    test_size   = int(1e4)

    # placeholder state
    candidate_state = np.zeros((1, grid.ndim))
    safe_level = 1.
    lagrange_multiplier = 1000
    #
    level_multiplier = 1.3,
    # level_multiplier = 1.1,
    learning_rate = 5e-3,
    batch_size    = int(1e3),

    ############################# training loop #############################


    exit()
    z = roa.reshape(grids.num_points)
    print('num_points', grids.num_points)
    ctrl.close()
    random_env.close()
    print('roa', z)
    
    if save_data:
        results = {'roa': roa, 'grids': grids, }
                # 'trajs_data': all_trajs, 'metrics': metrics \
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    # plot ROA of the 1st and 3rd state dimension
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=False)
    fig.subplots_adjust(wspace=0.35)
    x_max = np.max(np.abs(grids.all_points), axis=0)
    pos_max = x_max[0]
    theta_max = x_max[2]
    plot_limits = np.hstack((- np.rad2deg([pos_max, theta_max]), \
                               np.rad2deg([pos_max, theta_max])))
    # extract the 1st and 3rd state dimension of z
    print('z.shape', z.shape)
    z = z
    # print('\n')
    # print('z after extract', z)
    # print('z.shape', z.shape)
    exit()


    ax = plt.subplot(121)
    alpha = 1
    colors = [None] * 4
    colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
    colors[1] = (230/255, 159/255, 0)       # NN  - orange
    colors[2] = (0, 114/255, 178/255)       # LQR - blue
    colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow

    # True ROA
    ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
    ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)


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
