'''This script tests the RL-MPC implementation.'''

import os
import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Cost, Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(plot=True, training=False, n_episodes=1, n_steps=None, curr_path='.'):
    '''Main function to run RL-MPC experiments.

    Args:
        plot (bool): Whether to plot the results.
        training (bool): Whether to train the MPSC or load pre-trained values.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    system = config.task

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')

    # Run without safety filter
    experiment = BaseExperiment(env, ctrl)
    results, uncert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    elapsed_time_uncert = results['timestamp'][0][-1] - results['timestamp'][0][0]


if __name__ == '__main__':
    run()
