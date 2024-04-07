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

def run(gui=True, n_episodes=1, n_steps=None, save_data=False, data = None):
    # load the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../dev_pendulum/temp-data/')
    algo = 'linear_mpc'
    system = 'pendulum'
    task = 'stabilization'
    # L = 0.5
    # data_file = f'{algo}_data_{system}_{task}_{L}.pkl'
    # abs_data_path = os.path.join(data_dir, data_file)
    # print('abs_data_path: ', abs_data_path)
    # with open(abs_data_path, 'rb') as f:
    #     data = pickle.load(f)

    # L_small = 0.8
    # data_file_small = f'{algo}_data_{system}_{task}_{L_small}.pkl'
    # abs_data_path_small = os.path.join(data_dir, data_file_small)
    # print('abs_data_path_small: ', abs_data_path_small)
    # with open(abs_data_path_small, 'rb') as f:
    #     data_small = pickle.load(f)

    # L_large = 3.0
    # data_file_large = f'{algo}_data_{system}_{task}_{L_large}.pkl'
    # abs_data_path_large = os.path.join(data_dir, data_file_large)
    # print('abs_data_path_large: ', abs_data_path_large)
    # with open(abs_data_path_large, 'rb') as f:
    #     data_large = pickle.load(f)
    input_set_large = []
    input_set_small = []
    target_set_large = []
    target_set_small = []
    state_set_small = []
    state_set_large = []
    action_set_small = []
    action_set_large = []

    for i in range(2):
        idx = repr(i+1)
        data_file_small = f'{algo}_data_{system}_{task}_reach_{idx}.pkl'
        abs_data_path_small= os.path.join(data_dir, data_file_small)
        print('abs_data_path_small: ', abs_data_path_small)
        with open(abs_data_path_small, 'rb') as f:
            data_small = pickle.load(f)

        
        data_file_large = f'{algo}_data_{system}_{task}_not_{idx}.pkl'
        abs_data_path_large = os.path.join(data_dir, data_file_large)
        print('abs_data_path_large: ', abs_data_path_large)
        with open(abs_data_path_large, 'rb') as f:
            data_large = pickle.load(f)


        state_small = data_small['trajs_data']['obs'][0]
        action_small = data_small['trajs_data']['action'][0]
        input_small = data_small['train_inputs']
        target_small = data_small['train_targets'][:,-1]
        

        state_large = data_large['trajs_data']['obs'][0]
        action_large = data_large['trajs_data']['action'][0]
        input_large = data_large['train_inputs']
        target_large = data_large['train_targets'][:,-1]

        input_set_large.append(input_large)
        input_set_small.append(input_small)
        target_set_small.append(target_small)
        target_set_large.append(target_large)
        state_set_small.append(state_small)
        state_set_large.append(state_large)
        action_set_small.append(action_small)
        action_set_large.append(action_large)
        # Plot the state path
        plt.figure(1)
        # plt.plot(0, 0, 'ko')
        plt.plot(state_small[:, 0], state_small[:, 1],)
        plt.plot(state_large[:, 0], state_large[:, 1],)
        plt.plot(state_small[0, 0], state_small[0, 1], 'bx', markersize=12)
        plt.plot(state_large[0, 0], state_large[0, 1], 'bx', markersize=12)


        # plt.scatter(state_large_picked[:, 0], state_large_picked[:, 1], c='g', marker='*', label='L=3.0 (picked)')
        plt.xlabel('theta')
        plt.ylabel('theta_dot')


        # # plot the state and action trajectory in 3 x 1 subplots
        # fig, axs = plt.subplots(3, 1)
        # axs[0].plot(state_small[:, 0], 'r', label='stable')
        # axs[0].plot(state_large[:, 0], 'g', label='unstable')
        # axs[0].set(ylabel='theta')
        # axs[0].legend()
        # axs[1].plot(state_small[:, 1], 'r', label='stable')
        # axs[1].plot(state_large[:, 1], 'g', label='unstable')
        # axs[1].set(ylabel='theta_dot')
        # axs[1].legend()
        # axs[2].plot(action_small, 'r', label='stable')
        # axs[2].plot(action_large, 'g', label='unstable')
        # axs[2].set(ylabel='action')
        # axs[2].legend()


    input_large = np.concatenate(input_set_large, axis=0)
    input_small = np.concatenate(input_set_small, axis=0)
    target_large = np.concatenate(target_set_large, axis=0)
    target_small = np.concatenate(target_set_small, axis=0)
    state_large = np.concatenate(state_set_large, axis=0)
    num_data = input_small.shape[0]
    print('input_large.shape: ', input_large.shape)
    print('target_large.shape: ', target_large.shape)
    print('input_small.shape: ', input_small.shape)
    print('target_small.shape: ', target_small.shape)
    
    
    # sample num_data data points from the large dataset with a sequantial index
    idx = np.random.choice(np.arange(input_large.shape[0]), num_data, replace=False)
    idx = np.sort(idx)
    print('idx: ', idx)

    state_large_picked = state_large[idx]
    input_large = input_large[idx]
    target_large = target_large[idx]
    plt.figure(1)
    plt.plot(0, 0, 'ro', label='Goal', markersize=12)
    plt.scatter(state_large_picked[:, 0], state_large_picked[:, 1], \
                c='g', marker='*', label='unstable (picked)', s=100)
    plt.legend()

    plt.show()
    exit()
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
    data_small = {'train_input': train_input_small, 'train_target': train_target_small, 'test_input': test_input_small, 'test_target': test_target_small}
    data_large = {'train_input': train_input_large, 'train_target': train_target_large, 'test_input': test_input_large, 'test_target': test_target_large}

    # Plot the state path
    # plt.figure()
    # plt.plot(0, 0, 'ko',label='Goal')
    # plt.plot(state[0, 0], state[0, 1], 'bx', label='Start')
    # plt.plot(state[:, 0], state[:, 1], 'b', label='L=0.5 (true)')
    # plt.plot(state_small[:, 0], state_small[:, 1], 'r', label='L=0.8')
    # plt.plot(state_large[:, 0], state_large[:, 1], 'g', label='L=3.0')
    # plt.scatter(state_large_picked[:, 0], state_large_picked[:, 1], c='g', marker='*', label='L=3.0 (picked)')
    # plt.xlabel('theta')
    # plt.ylabel('theta_dot')
    # plt.legend()

    # # plot the state and action trajectory in 3 x 1 subplots
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(state[:, 0], 'b', label='L=0.5 (true)')
    # axs[0].plot(state_small[:, 0], 'r', label='L=0.8')
    # axs[0].plot(state_large[:, 0], 'g', label='L=3.0')
    # axs[0].set(ylabel='theta')
    # axs[0].legend()
    # axs[1].plot(state[:, 1], 'b', label='L=0.5 (true)')
    # axs[1].plot(state_small[:, 1], 'r', label='L=0.8')
    # axs[1].plot(state_large[:, 1], 'g', label='L=3.0')
    # axs[1].set(ylabel='theta_dot')
    # axs[1].legend()
    # axs[2].plot(action, 'b', label='L=0.5 (true)')
    # axs[2].plot(action_small, 'r', label='L=0.8')
    # axs[2].plot(action_large, 'g', label='L=3.0')
    # axs[2].set(ylabel='action')
    # axs[2].legend()

    # plt.show()

    model_type = ZeroMeanIndependentGPModel
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
    normalize = True

    n_iter = 1000
    learning_rate = 0.1
    gpu = False
    GP_name_small = 'GP_small.pth'
    GP_name_large = 'GP_large.pth'
    T = 100
    num_training_data = train_input_small.shape[0]
    round = T if T < num_training_data else num_training_data
    print('total round: ', round)
    MI_history_small = []
    MI_history_large = []
    for i in range(round):
        print('round: ', i)
        if i == 0:
            # pick one random data point from the small and large dataset with torch
            idx_small = torch.randint(0, train_input_small.shape[0], (1,))
            idx_large = torch.randint(0, train_input_large.shape[0], (1,))
            # select the input and target data and pop the selected data point
            selected_input_small = train_input_small[idx_small, :]
            selected_target_small = train_target_small[idx_small]
            selected_input_large = train_input_large[idx_large, :]
            selected_target_large = train_target_large[idx_large]
            train_input_small = torch.cat((train_input_small[:idx_small], train_input_small[idx_small+1:]), dim=0)
            train_target_small = torch.cat((train_target_small[:idx_small], train_target_small[idx_small+1:]), dim=0)
            train_input_large = torch.cat((train_input_large[:idx_large], train_input_large[idx_large+1:]), dim=0)
            train_target_large = torch.cat((train_target_large[:idx_large], train_target_large[idx_large+1:]), dim=0)

        else:
            # make prediction at every data point in the data set
            mean_small, cov_small, pred_small = GP_small.predict(train_input_small)
            mean_large, cov_large, pred_large = GP_large.predict(train_input_large)
            std_small = pred_small.stddev
            std_large = pred_large.stddev
            # pick the data point with the highest uncertainty
            idx_small = torch.argmax(std_small)
            idx_large = torch.argmax(std_large)
            # select the input and target data and pop the selected data point
            selected_input_small = torch.cat((selected_input_small, train_input_small[idx_small, :].unsqueeze(0)), dim=0)
            selected_target_small = torch.cat((selected_target_small, train_target_small[idx_small].unsqueeze(0)), dim=0)
            selected_input_large = torch.cat((selected_input_large, train_input_large[idx_large, :].unsqueeze(0)), dim=0)
            selected_target_large = torch.cat((selected_target_large, train_target_large[idx_large].unsqueeze(0)), dim=0)
            train_input_small = torch.cat((train_input_small[:idx_small], train_input_small[idx_small+1:]), dim=0)
            train_target_small = torch.cat((train_target_small[:idx_small], train_target_small[idx_small+1:]), dim=0)
            train_input_large = torch.cat((train_input_large[:idx_large], train_input_large[idx_large+1:]), dim=0)
            train_target_large = torch.cat((train_target_large[:idx_large], train_target_large[idx_large+1:]), dim=0)
            

        # init and train the GP model
        GP_small = GaussianProcess(model_type,
                            deepcopy(likelihood),
                            input_mask=[0, 1, 2],
                            normalize=normalize)
        GP_large = GaussianProcess(model_type,
                                    deepcopy(likelihood),
                                    input_mask=[0, 1, 2],
                                    normalize=normalize)
        GP_small.train(selected_input_small, selected_target_small, \
                        test_input_small, test_target_small, n_train=n_iter, 
                        learning_rate=learning_rate, gpu=gpu)
        GP_large.train(selected_input_large, selected_target_large, \
                        test_input_large, test_target_large, n_train=n_iter,\
                        learning_rate=learning_rate, gpu=gpu)
        # compute the mutual information
        # GP_small._compute_GP_covariances(train_x=selected_input_small)
        # GP_large._compute_GP_covariances(train_x=selected_input_large)
        print('selected_input_small.shape: ', selected_input_small.shape)
        print('selected_input_small.double().shape: ', selected_input_small.double().shape)
        # if i >= 1:
        # Gram_small = compute_gram_matrix(GP=GP_small, x=selected_input_small.double())
        # Gram_large = compute_gram_matrix(GP=GP_large, x=selected_input_large.double())
        # print('Gram_small.shape: ', Gram_small.shape)
        # exit()
        GP_small._compute_GP_covariances(train_x=selected_input_small)
        K_plus_noise_small = GP_small.model.K_plus_noise
        prior_std_small = GP_small.model.likelihood.noise
        MI_small = 0.5 * torch.logdet(K_plus_noise_small / prior_std_small**2)
        GP_large._compute_GP_covariances(train_x=selected_input_large)
        K_plus_noise_large = GP_large.model.K_plus_noise
        prior_std_large = GP_large.model.likelihood.noise
        MI_large = 0.5 * torch.logdet(K_plus_noise_large / prior_std_large**2)
        MI_history_small.append(MI_small.item())
        MI_history_large.append(MI_large.item())

    print('MI_history_small: ', MI_history_small)
    print('MI_history_large: ', MI_history_large)
    

    return MI_history_small, MI_history_large


def se_kernel(x1, x2, sigma=1.0, l=1.0):
    return sigma**2 * torch.exp(-0.5 * (x1 - x2)**2 / l**2)

def compute_gram_matrix(GP, x, kernel=se_kernel):
    '''
    Compute the Gram matrix for the input data x using the given kernel function.

    '''
    sigma = GP.model.covar_module.base_kernel.lengthscale.detach()
    l = GP.model.covar_module.outputscale.detach()
    n = x.shape[0]
    K = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            print('x[i, :]: ', x[i, :])
            print('x[i, :].shape: ', x[i, :].shape)
            print('x[j, :]: ', x[j, :])
            print('x[j, :].shape: ', x[j, :].shape)
            K[i, j] = kernel(x[i, :], x[j, :], sigma=sigma, l=l)
    return K

if __name__ == '__main__':
    plot_result = True
    # plot_result = False
    if plot_result:
        # load_result
        MI_collection_small = np.load('MI_collection_small_same.npy')
        MI_collection_large = np.load('MI_collection_large_same.npy')
        # print('MI_collection_small: ', MI_collection_small)
        # print('MI_collection_large: ', MI_collection_large)
        s = 2 # times of standard deviation
        MI_collection_small_mean = np.mean(MI_collection_small, axis=0)
        MI_collection_small_std = np.std(MI_collection_small, axis=0)
        MI_collection_large_mean = np.mean(MI_collection_large, axis=0)
        MI_collection_large_std = np.std(MI_collection_large, axis=0)
        # plot the result
        plt.figure()
        plt.plot(MI_collection_small_mean, 'r', label='stable data set')
        plt.fill_between(np.arange(len(MI_collection_small_mean)), MI_collection_small_mean - s * MI_collection_small_std, MI_collection_small_mean + s * MI_collection_small_std, color='r', alpha=0.2)
        plt.plot(MI_collection_large_mean, 'g', label='unstable data set')
        plt.fill_between(np.arange(len(MI_collection_large_mean)), MI_collection_large_mean - s * MI_collection_large_std, MI_collection_large_mean + s * MI_collection_large_std, color='g', alpha=0.2)
        plt.xlabel('round')
        plt.ylabel('MI')
        plt.title('Mutual Information in different data sets')
        plt.legend()
        plt.show()
    else:
        MI_collection_small = []
        MI_collection_large = []
        for i in range(5):
            MI_history_small, MI_history_large = run()
            MI_collection_small.append(MI_history_small)
            MI_collection_large.append(MI_history_large)
        MI_collection_small = np.array(MI_collection_small)
        MI_collection_large = np.array(MI_collection_large)
        print('MI_collection_small: ', MI_collection_small)
        print('MI_collection_large: ', MI_collection_large)
        np.save('MI_collection_small_same.npy', MI_collection_small)
        np.save('MI_collection_large_same.npy', MI_collection_large)