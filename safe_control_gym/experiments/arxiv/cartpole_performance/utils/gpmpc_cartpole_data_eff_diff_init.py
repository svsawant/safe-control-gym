"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

"""
import os

import munch
import yaml
import numpy as np
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config

# To set relative pathing of experiment imports.
import sys
import os.path as path
sys.path.append(path.abspath(path.join(__file__, "../../../utils/")))
from gpmpc_plotting_utils import make_plots, gather_training_samples, plot_data_eff_from_csv
from safe_control_gym.lyapunov.utilities import *
from safe_control_gym.lyapunov.lyapunov import *

def run(config, grids=None, sampled_init_state_idx=0):
    # read the prior parameters from the config
    M = config.algo_config.prior_info.prior_prop['cart_mass']
    m = config.algo_config.prior_info.prior_prop['pole_mass']
    l = config.algo_config.prior_info.prior_prop['pole_length']
    # if grids is not None:
    init_state = grids.index_to_state(sampled_init_state_idx).reshape(-1,1)    
    print('init_state:', init_state)
    
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    
    print(env_func.keywords)
    init_state_dict = {'init_x': init_state[0], 'init_x_dot': init_state[1], \
                        'init_theta': init_state[2], 'init_theta_dot': init_state[3]}
    print('init_state_dict:', init_state_dict)
    # env_func(init_state=init_state_dict)
    # exit()
    config.algo_config.output_dir = config.output_dir
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )
    ctrl.reset()

    num_epochs = config.num_epochs
    num_train_episodes_per_epoch = config.num_train_episodes_per_epoch
    num_test_episodes_per_epoch = config.num_test_episodes_per_epoch
    num_samples = config.num_samples # training data 
    train_runs = {0: {}}
    test_runs = {0: {}}
    
    # set up training and test environments, both with randomized initial states
    if config.same_train_initial_state:
        train_envs = []
        for epoch in range(num_epochs):
            train_envs.append(env_func(init_state=init_state_dict,randomized_init=False))
            train_envs[epoch].action_space.seed(config.seed)
    else:
        train_env = env_func(init_state=init_state_dict,randomized_init=False)
        train_env.action_space.seed(config.seed)
        train_envs = [train_env]*num_epochs
    #init_test_states = get_random_init_states(env_func, num_test_episodes_per_epoch)
    test_envs = []
    if config.same_test_initial_state:
        for epoch in range(num_epochs):
            test_envs.append(env_func(init_state=init_state_dict,randomized_init=False))
            test_envs[epoch].action_space.seed(config.seed)
    else:
        test_env = env_func(init_state=init_state_dict,randomized_init=False)
        test_env.action_space.seed(config.seed)
        test_envs = [test_env]*num_epochs

    # run prior policy for multiple episodes
    for episode in range(num_train_episodes_per_epoch):
        run_results = ctrl.prior_ctrl.run(env=train_envs[0],
                                          terminate_run_on_done=config.terminate_train_on_done)
        train_runs[0].update({episode: munch.munchify(run_results)})
        ctrl.reset()
    # test prior policy for multiple episodes
    for test_ep in range(num_test_episodes_per_epoch):
        run_results = ctrl.run(env=test_envs[0],
                               terminate_run_on_done=config.terminate_test_on_done)
        test_runs[0].update({test_ep: munch.munchify(run_results)})
    ctrl.reset()

    # training loop
    for epoch in range(1, num_epochs):
        # only take data from the last episode from the last epoch
        # and only gather num_samples data points (might be less by rounding)
        if config.rand_data_selection:
            x_seq, actions, x_next_seq = gather_training_samples(train_runs, epoch-1, num_samples, train_envs[epoch-1].np_random)
        else:
            x_seq, actions, x_next_seq = gather_training_samples(train_runs, epoch-1, num_samples)
        print(x_seq.shape, actions.shape, x_next_seq.shape)
        # print(num_epochs, num_train_episodes_per_epoch, num_test_episodes_per_epoch, num_samples, train_envs[epoch-1].state_dim, config.output_dir)
        train_inputs, train_outputs = ctrl.preprocess_training_data(x_seq, actions, x_next_seq)
        # print(train_inputs.shape, train_outputs.shape)
        # input('press enter to continue')
        # train policy
        _ = ctrl.learn(input_data=train_inputs, target_data=train_outputs)
        # input('press enter to continue')
        # Test new policy.
        test_runs[epoch] = {}
        for test_ep in range(num_test_episodes_per_epoch):
            ctrl.x_prev = test_runs[epoch-1][episode]['obs'][:ctrl.T+1,:].T
            ctrl.u_prev = test_runs[epoch-1][episode]['action'][:ctrl.T,:].T
            ctrl.reset()
            run_results = ctrl.run(env=test_envs[epoch],
                                   terminate_run_on_done=config.terminate_test_on_done)
            test_runs[epoch].update({test_ep: munch.munchify(run_results)})
        # gather training data
        train_runs[epoch] = {}
        for episode in range(num_train_episodes_per_epoch):
            ctrl.reset()
            ctrl.x_prev = train_runs[epoch-1][episode]['obs'][:ctrl.T+1,:].T
            ctrl.u_prev = train_runs[epoch-1][episode]['action'][:ctrl.T,:].T
            # once the goal is reached, the episode is terminated
            # actual amount of data collected may be less than num_samples?
            run_results = ctrl.run(env=train_envs[epoch],
                                   terminate_run_on_done=config.terminate_train_on_done)
            train_runs[epoch].update({episode: munch.munchify(run_results)})

        lengthscale, outputscale, noise, kern = ctrl.gaussian_process.get_hyperparameters(as_numpy=True)
        trajectory = 0
        # save result of each epoch
        data_file_name = 'data_epoch_{}_init_{}_M_{:0.1f}_m_{:0.1f}_l_{:0.1f}_prec_{}_{}_{}_{}'.format(epoch, sampled_init_state_idx, M, m, l, prec[0], prec[1], prec[2], prec[3])
        np.savez(os.path.join(config.output_dir, data_file_name),
                 train_runs=train_runs,
                 test_runs=test_runs,
                 num_epochs=num_epochs,
                 num_train_episodes_per_epoch=num_train_episodes_per_epoch,
                 num_test_episodes_per_epoch=num_test_episodes_per_epoch,
                 num_samples=num_samples,
                 trajectory=trajectory,
                 ctrl_freq=config.task_config.ctrl_freq,
                 lengthscales=lengthscale,
                 outputscale=outputscale,
                 noise=noise,
                 kern=kern,
                 train_data=ctrl.train_data,
                 test_data=ctrl.test_data,
                 data_inputs=ctrl.data_inputs,
                 data_targets=ctrl.data_targets)

        # make_plots(test_runs, train_runs, train_envs[0].state_dim, config.output_dir)

    # fname = os.path.join(config.output_dir, 'figs', 'avg_rmse_cost_learning_curve.csv')
    # plot_data_eff_from_csv(fname,
                        #    'Cartpole Data Efficiency')
    #plot_runs(test_runs, num_epochs)
    return train_runs, test_runs

if __name__ == "__main__":
    fac = ConfigFactory()
    fac.add_argument("--plot_dir", type=str, default='', help="Create plot from CSV file.")
    config = fac.merge()
    set_dir_from_config(config)
    mkdirs(config.output_dir)

    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)

    # fix numpy random seed for reproducibility
    np.random.seed(config.seed)
    # set up state space gridding
    dim_grid = 4
    grid_constraints = np.array([0.5, 1.5, 1.57, 1.57])
    grid_constraints = np.vstack((-1 * grid_constraints, \
                                        grid_constraints)).T

    # print('prior M: {}, m: {}, l: {}'.format(M, m, l))
    # exit()
    prec = [11, 31, 31, 31]
    grids = gridding(dim_grid, grid_constraints, prec)
    # sample random integers in the range of grids.nindex
    # and convert them to init states
    num_init_states = 30
    sampled_init_state_idx = np.random.choice(grids.nindex, num_init_states, replace=False)
    skip = -1
    
    print('sampled_init_state_idx:', sampled_init_state_idx)
    if config.plot_dir == '':
        for i in range(num_init_states):
            if i <= skip:
                continue
            else:#
                try:
                    train_runs, test_runs = run(config, grids=grids, sampled_init_state_idx=sampled_init_state_idx[i])
                except:
                    print('Error in run {} at idx {}'.format(i, sampled_init_state_idx[i]))
                    continue
    else:
        fname = config.plot_dir
        plot_data_eff_from_csv(fname,
                                 'Cartpole Data Efficiency')
