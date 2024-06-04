
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
import multiprocessing
import time
import json

# run each individual experiments
def run_each_exp(args):
    state_idx, config, grid, q = args
    #NOTE: if doesn't work, also create the controller here
    goal_reached = False

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    random_env = env_func(gui=False)
    init_state = grid.all_points[state_idx]
    init_state_dict = {'init_x': 0.0, 'init_x_dot': init_state[0], \
                        'init_theta': init_state[1], 'init_theta_dot': init_state[2]}
    init_state, _ = random_env.reset(init_state = init_state_dict)
    static_env = env_func(gui=False, random_state=False, init_state=init_state)
    static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
    # Create experiment, train, and run evaluation
    experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

    trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
    static_env.close()
    static_train_env.close()
    random_env.close()
    ctrl.close()

    goal_reached = trajs_data['info'][-1][-1]['goal_reached']
    # convert everything in trajs_data to list if it is a numpy array
    obs = trajs_data['obs'][0].tolist()
    action = trajs_data['action'][0].tolist()
    # merge trajectory data (dict) with state index and goal_reached
    # to a single dict
    result = {'idx': state_idx, 'goal_reached': goal_reached, \
                'obs': obs, 'action': action}
    # print('result', result)
    print('adding state idx {} to queue'.format(state_idx))
    q.put(result)
    time.sleep(2)
    return goal_reached

def run(gui=False, n_episodes=1, n_steps=None, save_data=False):
    '''The main function experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )
    # random_env = env_func(gui=False)
    # # print('random env\n', random_env.__dir__())
    # print('random env.POLE_MASS', random_env.POLE_MASS)
    # print('random env.EFFECTIVE_POLE_LENGTH', random_env.EFFECTIVE_POLE_LENGTH)
    # print('random env.CART_MASS', random_env.CART_MASS)
    # print('ctrl\n', ctrl.__dir__())
    # print('ctrl.model\n', ctrl.model.__dir__())
    # print('ctrl.model.pole_mass', ctrl.model.pole_mass) # m
    # print('ctrl.model.pole_length', ctrl.model.pole_length) # l
    # print('ctrl.model.cart_mass', ctrl.model.cart_mass) # M
    # prior knowledge of the system
    m = ctrl.model.pole_mass
    l = ctrl.model.pole_length
    M = ctrl.model.cart_mass
    dim_state = ctrl.model.x_sym.shape[0] # state dimension
    ctrl.close()

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes
    
    # state constraints
    # state_constraints = np.vstack((random_env.constraints.state_constraints[0].lower_bounds,
    #                                random_env.constraints.state_constraints[0].upper_bounds)).T
    # lower bound has the shape [-1, -1, -1, -1]
    
    
    # override state constraints for grids with self-defined constraints
    dim_grid = 3
    grid_constraints = np.array([3., 3.14, 3.14])
    # grid_constraints = np.array([1, 0.3, 0.2])
    grid_constraints = np.vstack((-1 * grid_constraints, \
                                        grid_constraints)).T
    # grid_constraints_ub = np.array([grid_constraints[0], -0.1, grid_constraints[2]])
    # grid_constraints = np.vstack((-1 * grid_constraints, \
    #                                     grid_constraints_ub)).T

    prec = [41, 51, 41]
    # prec = [2, 2, 2]
    grids = gridding(dim_grid, grid_constraints, prec)
    ################### parallel processing code here ###################
    # roa = compute_roa_fix_par(grids, env_func, ctrl, no_traj=True)
    # init result queue
    manager = multiprocessing.Manager()
    q = manager.Queue()
    # init pool
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count - 2)

    # get index list of all points in grids
    index_list = list(range(len(grids.all_points)))
    # print('index_list', index_list)
    # # get the state for the first index 
    # state = grids.all_points[index_list[0]]
    # print('state', state)
    result_file_name =  'results_M_{:0.1f}_m_{:0.1f}_l_{:0.1f}_prec_{}_{}_{}.json'\
                            .format(M, m, l, prec[0], prec[1], prec[2])
    print('result_file_name', result_file_name)
    # if the desired json file does not exist, create one
    if not os.path.exists(result_file_name):
        # write a empty list to json file
        result = []
        with open(result_file_name, 'w') as f:
            json.dump(result, f, indent=4)
        target_list = index_list
    elif os.path.exists(result_file_name):
        '''
        if the desired json file exists,
        (possibly because the program is 
        terminated before all tasks are done)
        '''
        # read the result from json file
        with open(result_file_name, 'r') as f:
            result = json.load(f)
           
        # extract the existing number list
        exist_idx_list = [x['idx'] for x in result]
        # subtract the existing number list from the desired number list
        target_list = list(set(index_list) - set(exist_idx_list))
        
    print('target_list', target_list)
    # # create a log file to record the progress
    # log_file_name = 'progress.json'
    # if not os.path.exists(log_file_name):
    #     # write a empty list to json file
    #     log = []
    #     with open(log_file_name, 'w') as f:
    #         json.dump(log, f, indent=4)
    # send task to pool
    # roa = pool.map_async(run_each_exp, [(idx, env_func, ctrl, grids, q) \
    #                                             for idx in target_list])
    print('config', config)
    print('grids', grids)
    print('q', q)
    roa = pool.map_async(run_each_exp, [(idx, config, grids, q) for idx in target_list])
    time.sleep(5)
    all_done = False
    ###################  grab results from queue ###################
    time_before = time.time()
    while not all_done:
        # check if all tasks are done
        with open(result_file_name, 'r') as f:
            result = json.load(f)
        # Grab all finished results from the queue
        while q.qsize() > 0:
            result.append(q.get())
        # save results to disk
        with open(result_file_name, "w") as f:
            json.dump(result, f, indent=4)
        # terminate the pool when all tasks are done
        if len(result) == len(index_list):
            all_done = True
        else:
            current_time = time.time()
            # print the time elapsed every 30 seconds
            if (current_time - time_before) % 30 == 0:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('time elapsed', current_time - time_before)
                # print the number of tasks done
                print('progress: {}, {}'.format(len(result), len(index_list)))
                # with open(log_file_name, 'w') as f:
                #     json.dump(len(result)/len(index_list), f, indent=4)
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    time_after = time.time()
    print('time for parallel processing', time_after - time_before)
    print('roa', roa.get())
    roa = np.array(roa.get())

    #####################################################################
    
    # sort the result json file
    # sort_result = False
    sort_result = True
    if sort_result:
        # read the result from json file
        with open(result_file_name, 'r') as f:
            result = json.load(f)
        # print('result before sorting', result)
        # sort the result list
        result.sort(key=lambda x: x['idx'])
        # print('result after sorting', result)
        # save the sorted result to json file
        with open(result_file_name, 'w') as f:
            json.dump(result, f, indent=4)
    
    # assemble roa
    roa = np.zeros(len(grids.all_points))
    for k, result in enumerate(result):
        roa[result['idx']] = result['goal_reached']

    # check results
    safe_fraction = np.sum(roa) / len(grids.all_points)
    print('safe fraction', safe_fraction)
    # concatenate all points in grids with roa
    res = np.hstack(( roa.reshape(-1, 1), grids.all_points))
    
    print('res\n', res)

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
