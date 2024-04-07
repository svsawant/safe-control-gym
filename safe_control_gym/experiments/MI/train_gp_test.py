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

import gpytorch
import torch

from safe_control_gym.controllers.mpc.gp_utils import ZeroMeanIndependentGPModel, \
                                                      GaussianProcess
from copy import deepcopy
from sklearn.model_selection import train_test_split

seed = 1

# load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../dev_pendulum/temp-data/')
algo = 'linear_mpc'
system = 'pendulum'
task = 'stabilization'
L = 0.5
data_file = f'{algo}_data_{system}_{task}_{L}.pkl'
abs_data_path = os.path.join(data_dir, data_file)
print('abs_data_path: ', abs_data_path)
with open(abs_data_path, 'rb') as f:
    data = pickle.load(f)

L_small = 0.8
data_file_small = f'{algo}_data_{system}_{task}_{L_small}.pkl'
abs_data_path_small = os.path.join(data_dir, data_file_small)
print('abs_data_path_small: ', abs_data_path_small)
with open(abs_data_path_small, 'rb') as f:
    data_small = pickle.load(f)

L_large = 3.0
data_file_large = f'{algo}_data_{system}_{task}_{L_large}.pkl'
abs_data_path_large = os.path.join(data_dir, data_file_large)
print('abs_data_path_large: ', abs_data_path_large)
with open(abs_data_path_large, 'rb') as f:
    data_large = pickle.load(f)

print('data.keys(): ', data.keys())
print('data[trajs_data].keys()', data['trajs_data'].keys())
# print('data[trajs_data][obs]', data['trajs_data']['obs'])
state = data['trajs_data']['obs'][0]
action = data['trajs_data']['action'][0]

state_small = data_small['trajs_data']['obs'][0]
action_small = data_small['trajs_data']['action'][0]
input_small = data_small['train_inputs']
target_small = data_small['train_targets'][:,-1]
num_data = input_small.shape[0]


state_large = data_large['trajs_data']['obs'][0]
action_large = data_large['trajs_data']['action'][0]
input_large = data_large['train_inputs']
target_large = data_large['train_targets'][:,-1]
# sample num_data data points from the large dataset with a sequantial index
idx = np.random.choice(np.arange(input_large.shape[0]), num_data, replace=False)
idx = np.sort(idx)
print('idx: ', idx)

state_large_picked = state_large[idx]
input_large = input_large[idx]
target_large = target_large[idx]

# split the data into training and testing
def split_data(input_data, target_data, split_ratio=0.8, gpu=False, seed=1):
    num_input_data = input_data.shape[0]
    train_idx, test_idx = train_test_split(np.arange(num_input_data), \
                                                            test_size=1-split_ratio, \
                                                            random_state=seed)
    train_input = input_data[train_idx]
    train_target = target_data[train_idx]
    test_input = input_data[test_idx]
    test_target = target_data[test_idx]
    if gpu:
        train_input = torch.from_numpy(train_input).cuda()
        train_target = torch.from_numpy(train_target).cuda()
        test_input = torch.from_numpy(test_input).cuda()
        test_target = torch.from_numpy(test_target).cuda()
    else:
        train_input = torch.from_numpy(train_input)
        train_target = torch.from_numpy(train_target)
        test_input = torch.from_numpy(test_input)
        test_target = torch.from_numpy(test_target)

    return train_input, train_target, test_input, test_target

train_input_small, train_target_small, test_input_small, test_target_small = split_data(input_small, target_small, seed=seed)
train_input_large, train_target_large, test_input_large, test_target_large = split_data(input_large, target_large, seed=seed)

# Plot the state path
plt.figure()
plt.plot(0, 0, 'ko',label='Goal')
plt.plot(state[0, 0], state[0, 1], 'bx', label='Start')
plt.plot(state[:, 0], state[:, 1], 'b', label='L=0.5 (true)')
plt.plot(state_small[:, 0], state_small[:, 1], 'r', label='L=0.8')
plt.plot(state_large[:, 0], state_large[:, 1], 'g', label='L=3.0')
plt.scatter(state_large_picked[:, 0], state_large_picked[:, 1], c='g', marker='*', label='L=3.0 (picked)')
plt.xlabel('theta')
plt.ylabel('theta_dot')
plt.legend()

# plot the state and action trajectory in 3 x 1 subplots
fig, axs = plt.subplots(3, 1)
axs[0].plot(state[:, 0], 'b', label='L=0.5 (true)')
axs[0].plot(state_small[:, 0], 'r', label='L=0.8')
axs[0].plot(state_large[:, 0], 'g', label='L=3.0')
axs[0].set(ylabel='theta')
axs[0].legend()
axs[1].plot(state[:, 1], 'b', label='L=0.5 (true)')
axs[1].plot(state_small[:, 1], 'r', label='L=0.8')
axs[1].plot(state_large[:, 1], 'g', label='L=3.0')
axs[1].set(ylabel='theta_dot')
axs[1].legend()
axs[2].plot(action, 'b', label='L=0.5 (true)')
axs[2].plot(action_small, 'r', label='L=0.8')
axs[2].plot(action_large, 'g', label='L=3.0')
axs[2].set(ylabel='action')
axs[2].legend()

# plt.show()

def run(gui=True, n_episodes=1, n_steps=None, save_data=False):
    model_type = ZeroMeanIndependentGPModel
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
    normalize = True
    GP_small = GaussianProcess(model_type,
                               deepcopy(likelihood),
                               input_mask=[0],
                               normalize=normalize)
    GP_large = GaussianProcess(model_type,
                                deepcopy(likelihood),
                                input_mask=[0],
                                normalize=normalize)
    n_iter = 1000
    learning_rate = 0.1
    gpu = False
    GP_name_small = 'GP_small.pth'
    GP_name_large = 'GP_large.pth'
    abs_GP_path = os.path.join(current_dir, GP_name_small)
    print('abs_GP_path: ', abs_GP_path)
    print('os.path.exists(abs_GP_path): ', os.path.exists(abs_GP_path))
    # exit()
    if os.path.exists(os.path.join(current_dir, GP_name_small)) and \
        os.path.exists(os.path.join(current_dir, GP_name_large)):
        print('===== Loading the trained GP model =====')
        GP_small.init_with_hyperparam(train_input_small, train_target_small, 
                                      path_to_statedict=GP_name_small)
        GP_large.init_with_hyperparam(train_input_large, train_target_large, 
                                      path_to_statedict=GP_name_large)
    else:
        print('===== Training the GP model =====')
        GP_small.train(train_input_small, train_target_small,\
                        test_input_small, test_target_small,\
                        n_train=n_iter, learning_rate=learning_rate, gpu=gpu,
                        fname='GP_small.pth')
        # GP_small.plot_trained_gp(input_small, target_small, output_label='small', fig_count=1)
        GP_large.train(train_input_large, train_target_large,\
                            test_input_large, test_target_large,\
                            n_train=n_iter, learning_rate=learning_rate, gpu=gpu,
                            fname='GP_large.pth')
        # GP_large.plot_trained_gp(input_large, target_large, output_label='large', fig_count=2)
    
    
    plt.show()
    return 1



if __name__ == '__main__':
    run()
