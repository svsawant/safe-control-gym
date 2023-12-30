
import numpy as np
from matplotlib.colors import ListedColormap
from lyapnov import GridWorld
from safe_control_gym.experiments.base_experiment import BaseExperiment

def gridding(state_dim, state_constraints, num_states = 251, use_zero_threshold = True):
    ''' evenly discretize the state space

    Args:
        state_dim (int): The dimension of the state space.
        state_constraints (np array): The constraints of the state space.
        num_state (int): The number of states along each dimension.
        use_zero_threshold (bool): Whether to use zero threshold.
                                   False: the grid is infinitesimal
    '''
    
    # State grid
    if state_constraints is None:
        state_constraints = np.array([[-1., 1.], ] * state_dim)
    grid_limits = state_constraints
    state_discretization = GridWorld(grid_limits, num_states)

    # Discretization constant
    if use_zero_threshold:
        tau = 0.0 # assume the grid is infinitesimal
    else:
        tau = np.sum(state_discretization.unit_maxes) / 2

    print('Grid size: {}'.format(state_discretization.nindex))
    print('Discretization constant (tau): {}'.format(tau))
    return state_discretization

def compute_roa(grid, env_func, ctrl ,equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex # number of points in the discretization
        ndim = grid.ndim  # dimension of the state space

    # Forward-simulate all trajectories from initial points in the discretization
    # if no_traj:
    #     end_states = all_points
    #     for t in range(1, horizon):
    #         end_states = closed_loop_dynamics(end_states)
    # else:
    #     trajectories = np.empty((nindex, ndim, horizon))
    #     trajectories[:, :, 0] = all_points
    #     for t in range(1, horizon):
    #         trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
    #     end_states = trajectories[:, :, -1]
    ramdom_env = env_func(gui=False)

    roa = np.zeros((nindex))
    
    # if no_traj:
    #     pass
    # else:
    #     trajectories = np.empty((nindex, ndim, horizon))
    for state_index in range(nindex):
        # for all initial state in the grid
        print('state_index', state_index)
        init_state = grid.all_points[state_index]
        init_state_dict = {'init_x': init_state[0], 'init_x_dot': init_state[1], \
                            'init_theta': init_state[2], 'init_theta_dot': init_state[3]}
        init_state, _ = ramdom_env.reset(init_state = init_state_dict)
        print('init_state', init_state)
        static_env = env_func(gui=False, random_state=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)
        # Create experiment, train, and run evaluation
        experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)

        trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1, verbose=False)
        print('obs\n', trajs_data['obs'])
        print('trajs_data\n', trajs_data['info'][-1][-1])
        print('\n')
        print('trajs_data[\'info\']\n', trajs_data['info'][-1][-1]['goal_reached'])
        # input('press enter to continue')
        print('\n')
        # exit()
        # print('goal reached', trajs_data['info'][-1][1]['goal_reached'])
        roa[state_index] = trajs_data['info'][-1][-1]['goal_reached']
        # close environments
        static_env.close()
        static_train_env.close()

    # if equilibrium is None:
    #     equilibrium = np.zeros((1, ndim))

    # # Compute an approximate ROA as all states that end up "close" to 0
    # dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    # roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories
    




def binary_cmap(color='red', alpha=1.):
    """Construct a binary colormap."""
    if color == 'red':
        color_code = (1., 0., 0., alpha)
    elif color == 'green':
        color_code = (0., 1., 0., alpha)
    elif color == 'blue':
        color_code = (0., 0., 1., alpha)
    else:
        color_code = color
    transparent_code = (1., 1., 1., 0.)
    return ListedColormap([transparent_code, color_code])
