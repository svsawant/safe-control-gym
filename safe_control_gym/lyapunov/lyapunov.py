
from collections.abc import Sequence
import itertools

import numpy as np
import torch

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Add the configuration settings
class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = torch.float32
        #######################################################################
        # Batch size for stability verification
        # TODO: change this back to 10000 in the future (by Mingxuan)
        self.gp_batch_size = 100 # originally 10000
        #######################################################################

    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        return np.float32

    def __repr__(self):
        """Print the parameters."""
        params = ['Configuration parameters:', '']
        for param, value in self.__dict__.items():
            params.append('{}: {}'.format(param, value.__repr__()))

        return '\n'.join(params)

config = Configuration()
del Configuration
_EPS = np.finfo(config.np_dtype).eps

class DimensionError(Exception):
    pass

class GridWorld(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    NOTE: in original Lyapunov NN, the grid is defined in a normalized 
          fashion (i.e. [-1, 1] for each dimension)
    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int16, copy=False)
        self.state_dim = len(self.limits)
        # print('self.state_dim: ', self.state_dim)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)

        self.ndim = len(self.limits)
        self._all_points = None

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            # my own implementation
            mesh = np.stack(np.meshgrid(*self.discrete_points),-1).reshape(-1,self.state_dim)
            self._all_points = mesh.astype(config.np_dtype)
            # if self.all_points.shape[1] == 2:
                # swap the first two columns
                # self._all_points[:,[0,1]] = self._all_points[:,[1,0]]

            # original implementation
            # mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            # points = np.column_stack(col.ravel() for col in mesh)
            # each row of the mesh is a point in the stat space
            # self._all_points = points.astype(config.np_dtype)

        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)

class QuadraticFunction(object):
    """A quadratic function.

    values(x) = x.T P x

    Parameters
    ----------
    matrix : np.array
        2d cost matrix for lyapunov function.

    """
    def __init__(self, matrix):
        """Initialization, see `QuadraticLyapunovFunction`."""
        super(QuadraticFunction, self).__init__()

        self.matrix = np.atleast_2d(matrix).astype(config.np_dtype)
        # print('self.matrix\n',self.matrix)
        self.ndim = self.matrix.shape[0]
        # with tf.variable_scope(self.scope_name):
        #     self.matrix = tf.Variable(self.matrix)

    def __call__(self, *args, **kwargs):
        """Evaluate the function using the template to ensure variable sharing.

        Parameters
        ----------
        args : list
            The input arguments to the function.
        kwargs : dict, optional
            The keyword arguments to the function.

        Returns
        -------
        outputs : list
            The output arguments of the function as given by evaluate.

        """
        
        outputs = self.forward(*args, **kwargs)
        return outputs
    
    def forward(self, points):
        """Like evaluate, but returns a tensor instead."""
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        # linear_form = tf.matmul(points, self.matrix)
        # print('points\n', points)
        # print('points shape\n', points.shape)
        # print('points type\n', type(points))
        # convert points to np array
        if isinstance(points, torch.Tensor):
            # if the tensor is on GPU, convert it to CPU first
            if points.is_cuda:
                points = points.cpu()
            points = points.detach().numpy()
            points = np.reshape(points, [-1])
        # print('points\n', points)
        # reshape points to 1d array  
        
        linear_form = points @ self.matrix
        quadratic = linear_form @ points.T
        # return tf.reduce_sum(quadratic, axis=1, keepdims=True)
        # print('quadratic\n',quadratic)
        return torch.tensor(quadratic)

    def gradient(self, points):
        """Return the gradient of the function."""
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()
        # return tf.matmul(points, self.matrix + self.matrix.T)
        return torch.matmul(torch.tensor(points, dtype=config.dtype), \
                            torch.tensor(self.matrix + self.matrix.T, dtype=config.dtype))

class LyapunovNN(torch.nn.Module):
    # def __init__(self, dim_input, layer_dims, activations):
    def __init__(self, input_dim, layer_dims, activations, eps=1e-6, device='cpu'):
        super(LyapunovNN, self).__init__()
        # network layers
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.eps = eps
        self.layers = torch.nn.ModuleList()
        self.kernel = []
        self.device = device

        if layer_dims[0] < input_dim:
            raise ValueError('The first layer dimension must be at \
                             least the input dimension!')

        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase \
                             the dimension of its input!')

        self.hidden_dims = np.zeros(self.num_layers, dtype=int)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)

    #     # build the nn structure
    #     self.linear1 = torch.nn.Linear(2, 2, bias=False)
    #     self.linear2 = torch.nn.Linear(2, 62, bias=False)
    #     self.linear3 = torch.nn.Linear(64, 33, bias=False)
    #     self.linear4 = torch.nn.Linear(64, 33, bias=False)
    #     W1 = self.linear1.weight
    #     W2 = self.linear2.weight
    #     # print('W1.shape\n', W1.shape)
    #     # print('W2.shape\n', W2.shape)
    #     inter_kernel = torch.matmul(W1.T, W1) + self.eps * torch.eye(W1.shape[1])
    #     self.kernel_1 = torch.cat((inter_kernel, W2), dim=0)
    #     W3 = self.linear3.weight
    #     self.kernel_2 = torch.matmul(W3.T, W3) + self.eps * torch.eye(W3.shape[1])
    #     W4 = self.linear4.weight
    #     self.kernel_3 = torch.matmul(W4.T, W4) + self.eps * torch.eye(W4.shape[1])

    # def forward(self, x):
    #     if isinstance(x, np.ndarray):
    #         x = torch.from_numpy(x).float()
    #     x = self.activations[0](torch.matmul(self.kernel_1, x))
    #     x = self.activations[1](torch.matmul(self.kernel_2, x))
    #     x = self.activations[2](torch.matmul(self.kernel_3, x))
    #     x = torch.sum(torch.square(x))
    #     return x
    
    # def update_kernel(self):
    #     # update the kernel
    #     W1 = self.linear1.weight
    #     W2 = self.linear2.weight
    #     inter_kernel = torch.matmul(W1.T, W1) + self.eps * torch.eye(W1.shape[1])
    #     self.kernel_1 = torch.cat((inter_kernel, W2), dim=0)
    #     W3 = self.linear3.weight
    #     self.kernel_2 = torch.matmul(W3.T, W3) + self.eps * torch.eye(W3.shape[1])
    #     W4 = self.linear4.weight
    #     self.kernel_3 = torch.matmul(W4.T, W4) + self.eps * torch.eye(W4.shape[1])
        

        # build the nn structure
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.layers.append(\
                        torch.nn.Linear(layer_input_dim, self.hidden_dims[i], bias=False))
            # W = self.layers[-1].weight
            # weight = W.clone()
            # weight = W
            # kernel = torch.matmul(weight.T, weight) + self.eps * torch.eye(W.shape[1])
            # kernel = torch.matmul(W.T, W) + self.eps * torch.eye(W.shape[1])
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                self.layers.append(torch.nn.Linear(layer_input_dim, dim_diff, bias=False))
                # print(kernel.shape, self.layers[-1].weight.shape)
                # kernel = torch.cat((kernel, self.layers[-1].weight), dim=0)
            # self.kernel.append(kernel)
        self.update_kernel()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        # put the input to the device
        x = x.to(self.device)
        
        for i in range(self.num_layers):
            # print('self.kernel[i].is_cuda\n', self.kernel[i].is_cuda)
            # print('x.is_cuda\n', x.is_cuda)
            layer_output = torch.matmul(self.kernel[i], x)
            x = self.activations[i](layer_output)
        values = torch.sum(torch.square(x), dim=-1)
        return values
    
    def update_kernel(self):
        self.kernel = [] # clear the kernel
        param_idx = 0 # for skipping the extra layer parameters
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            # build the positive definite part of the kernel
            W = self.layers[i + param_idx].weight
            weight = W.clone()
            kernel = torch.matmul(weight.T, weight) + self.eps * torch.eye(W.shape[1])
            # if the kernel need extra part, append the parameters of the next layer
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                kernel = torch.cat((kernel, self.layers[i+1].weight), dim=0)
                param_idx += 1
            # print('i: ', i)
            self.kernel.append(kernel)

    # def print_manual_kernel(self):
    #     print('Kernel 1:\n', self.kernel_1)
    #     print('Kernel 2:\n', self.kernel_2)
    #     print('Kernel 3:\n', self.kernel_3)

    #     # print kernel eigenvalues
    #     eigvals, _ = np.linalg.eig(self.kernel_1[0:2, :].detach().numpy())
    #     print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')
    #     eigvals, _ = np.linalg.eig(self.kernel_2.detach().numpy())
    #     print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')
    #     eigvals, _ = np.linalg.eig(self.kernel_3.detach().numpy())
    #     print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')

    def print_params(self):
        offset = 0
        # get nn parameters
        params = []
        for _, param in self.named_parameters():
            params.append(param.data)
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            print('Layer weights {}:'.format(i))
            W0 = params[offset + i]
            print('W0:\n{}'.format(W0))
            if dim_diff > 0:
                W1 = params[offset + 1 + i]
                print('W1:\n{}'.format(W1))
            else:
                offset += 1
            kernel = W0.T.dot(W0) + self.eps * np.eye(W0.shape[1])
            eigvals, _ = np.linalg.eig(kernel)
            print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')

class Lyapunov(object):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : callable or instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function.
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    lipschitz_dynamics : ndarray or float
        The Lipschitz constant of the dynamics. Either globally, or locally
        for each point in the discretization (within a radius given by the
        discretization constant. This is the closed-loop Lipschitz constant
        including the policy!
    lipschitz_lyapunov : ndarray or float
        The Lipschitz constant of the lyapunov function. Either globally, or
        locally for each point in the discretization (within a radius given by
        the discretization constant.
    tau : float
        The discretization constant.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    adaptive : bool, optional
        A boolean determining whether an adaptive discretization is used for
        stability verification.

    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov,
                 tau, policy, initial_set=None, adaptive=False):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()

        self.discretization = discretization
        self.policy = policy

        # Keep track of the safe sets
        self.safe_set = np.zeros(np.prod(discretization.num_points),
                                 dtype=bool)

        self.initial_safe_set = initial_set
        if initial_set is not None:
            # print('initial safe set\n', initial_set)
            # print('initial safe set shape\n', initial_set.shape)
            # print('initial safe set type\n', type(initial_set))
            # print('self.safe_set\n', self.safe_set)
            # print('self.safe_set shape\n', self.safe_set.shape)
            # print('self.safe_set type\n', type(self.safe_set))
            self.safe_set[initial_set] = True

        # Discretization constant
        self.tau = tau

        # Make sure dynamics are of standard framework
        self.dynamics = dynamics

        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function

        # Storage for graph
        self._storage = dict()
        # self.feed_dict = get_feed_dict(tf.get_default_graph())

        # Lyapunov values
        self.values = None

        # self.c_max = tf.placeholder(config.dtype, shape=())
        self.c_max = None
        # self.feed_dict[self.c_max] = 0.

        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov

        self.update_values()

        self.adaptive = adaptive

        # Keep track of the refinement `N(x)` used around each state `x` in
        # the adaptive discretization; `N(x) = 0` by convention if `x` is
        # unsafe
        self._refinement = np.zeros(discretization.nindex, dtype=int)
        if initial_set is not None:
            self._refinement[initial_set] = 1

    def update_values(self):
        """Update the discretized values when the Lyapunov function changes."""
        values = np.zeros(self.discretization.nindex)
        for i in range(self.discretization.nindex):
            # print('self.discretization.all_points[i]\n', self.discretization.all_points[i])
            # print('self.lyapunov_function(self.discretization.all_points[i]).squeeze(), \n', \
            #                     self.lyapunov_function(\
            #                     self.discretization.all_points[i]).squeeze())
            values[i] = self.lyapunov_function(\
                             self.discretization.all_points[i]).squeeze()
        self.values = values

    def update_safe_set(self, can_shrink=True, max_refinement=1,
                        safety_factor=1., parallel_iterations=1):
        """Compute and update the safe set.

        Parameters
        ----------
        can_shrink : bool, optional
            A boolean determining whether previously safe states other than the
            initial safe set must be verified again (i.e., can the safe set
            shrink in volume?)
        max_refinement : int, optional
            The maximum integer divisor used for adaptive discretization.
        safety_factor : float, optional
            A multiplicative factor greater than 1 used to conservatively
            estimate the required adaptive discretization.
        parallel_iterations : int, optional
            The number of parallel iterations to use for safety verification in
            the adaptive case. Passed to `tf.map_fn`.

        """
        safety_factor = np.maximum(safety_factor, 1.)

        np_states = lambda x: np.array(x, dtype=config.dtype)
        # decrease = lambda x: self.v_decrease_bound(x, self.dynamics(x, self.policy(x)))
        decrease = lambda x: self.v_decrease_bound(x, self.dynamics(x))
        threshold = lambda x: self.threshold(x, self.tau)
        np_negative = lambda x: np.squeeze(decrease(x) < threshold(x), axis=0)

        if can_shrink:
            # Reset the safe set and adaptive discretization
            safe_set = np.zeros_like(self.safe_set, dtype=bool)
            refinement = np.zeros_like(self._refinement, dtype=int)
            if self.initial_safe_set is not None:
                safe_set[self.initial_safe_set] = True
                refinement[self.initial_safe_set] = 1
        else:
            # Assume safe set cannot shrink
            safe_set = self.safe_set
            refinement = self._refinement

        value_order = np.argsort(self.values)
        safe_set = safe_set[value_order]
        refinement = refinement[value_order]

        # Verify safety in batches
        batch_size = config.gp_batch_size
        batch_generator = batchify((value_order, safe_set, refinement),
                                   batch_size)
        # print('batch_generator\n', batch_generator.__dir__())
        # exit()
        index_to_state = self.discretization.index_to_state

        #######################################################################

        for i, (indices, safe_batch, refine_batch) in batch_generator:
            # print('indices\n', indices)
            # print('safe_batch\n', safe_batch)
            # print('refine_batch\n', refine_batch)
            # exit()

            states = index_to_state(indices)
            np_state = np.squeeze(states)
            # print('np_states in update safe set\n', np_state)
            # print('np_states shape\n', np_state.shape)
            # print('np_states type\n', type(np_state))

            # Update the safety with the safe_batch result
            # negative = tf_negative.eval(feed_dict)
            # negative = np_negative(np_state)
            negative = np.zeros_like(safe_batch, dtype=bool)
            for state_index in range(len(np_state)):
                negative[state_index] = np_negative(np_state[state_index])
            # convert negative to np array
            negative = np.array(negative, dtype=bool)
            # check data type
            # print('negative\n', negative)
            # print('negative shape\n', negative.shape)
            # print('negative type\n', type(negative))
            # print('safe_batch\n', safe_batch)
            # print('safe_batch shape\n', safe_batch.shape)
            # print('safe_batch type\n', type(safe_batch))
            safe_batch |= negative
            # exit()
            refine_batch[negative] = 1

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)
            refine_bound = 0

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                    safe_batch[bound:] = False
                    refine_batch[bound:] = 0
                    break

        # The largest index of a safe value
        max_index = i + bound + refine_bound - 1

        #######################################################################

        # Set placeholder for c_max to the corresponding value
        self.c_max = self.values[value_order[max_index]]

        # Restore the order of the safe set and adaptive refinement
        safe_nodes = value_order[safe_set]
        self.safe_set[:] = False
        self.safe_set[safe_nodes] = True
        self._refinement[value_order] = refinement

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True
            self._refinement[self.initial_safe_set] = 1
        
    def threshold(self, states, tau=None):
        """Return the safety threshold for the Lyapunov condition.

        Parameters
        ----------
        states : ndarray or Tensor

        tau : float or Tensor, optional
            Discretization constant to consider.

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.

        """
        if tau is None:
            tau = self.tau
        # if state is not a tensor, convert it to a tensor
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=config.dtype, requires_grad=True)
            states = states.float()
        # print('states\n', states)
        lv = self._lipschitz_lyapunov(states)
        # print('lv\n', lv)
        # print('lv shape\n', lv.shape)
        # print('hasattr(self._lipschitz_lyapunov, __call__)\n', hasattr(self._lipschitz_lyapunov, '__call__'))
        ## TODO: check this part (by Mingxuan)
        # if hasattr(self._lipschitz_lyapunov, '__call__') and lv.shape[1] > 1:
        #     # lv = tf.norm(lv, ord=1, axis=1, keepdims=True)
        #     lv = torch.norm(lv, p=1, dim=1, keepdim=True)
        # convert states to np array
        if states.is_cuda:
            states = states.cpu()
        states = states.detach().numpy()
        lf = self._lipschitz_dynamics(states)
        return - lv * (1. + lf) * tau
    
    def v_decrease_bound(self, states, next_states):
        """Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array or tuple
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        upper_bound : np.array
            The upper bound on the change in values at each grid point.

        """
        v_dot, v_dot_error = self.v_decrease_confidence(states, next_states)

        return v_dot + v_dot_error
    
    def v_decrease_confidence(self, states, next_states):
        """Compute confidence intervals for the decrease along Lyapunov function.

        Parameters
        ----------
        states : np.array
            The states at which to start (could be equal to discretization).
        next_states : np.array
            The dynamics evaluated at each point on the discretization. If
            the dynamics are uncertain then next_states is a tuple with mean
            and error bounds.

        Returns
        -------
        mean : np.array
            The expected decrease in values at each grid point.
        error_bounds : np.array
            The error bounds for the decrease at each grid point

        """
        if isinstance(next_states, Sequence):
            next_states, error_bounds = next_states
            lv = self._lipschitz_lyapunov(next_states)
            # bound = tf.reduce_sum(lv * error_bounds, axis=1, keepdims=True)
            # bound = torch.sum(lv * error_bounds, dim=1, keepdim=True)
            bound = np.sum(lv * error_bounds, axis=1, keepdims=True)
        else:
            # bound = tf.constant(0., dtype=config.dtype)
            bound = torch.tensor(0., dtype=config.dtype)
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float64)
            states = states.float() # avoid feedforward data type error
        # next_states is of type casadi.DM
        # convert the next_states first to numpy array, then to torch tensor
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.tensor(np.array(next_states), dtype=torch.float64)
            next_states = next_states.float() # avoid feedforward data type error
        # print('next_states\n', next_states)
        # print('next_states shape\n', next_states.shape)
        # print('next_states type\n', type(next_states))
        # print('next_states data type\n', next_states.dtype)
        v_decrease = (self.lyapunov_function(next_states)
                      - self.lyapunov_function(states))

        return v_decrease, bound

# TODO: put this in a separate file (by Mingxuan)
def batchify(arrays, batch_size):
    """Yield the arrays in batches and in order.

    The last batch might be smaller than batch_size.

    Parameters
    ----------
    arrays : list of ndarray
        The arrays that we want to convert to batches.
    batch_size : int
        The size of each individual batch.
    """
    if not isinstance(arrays, (list, tuple)):
        arrays = (arrays,)

    # Iterate over array in batches
    for i, i_next in zip(itertools.count(start=0, step=batch_size),
                         itertools.count(start=batch_size, step=batch_size)):

        batches = [array[i:i_next] for array in arrays]

        # Break if there are no points left
        if batches[0].size:
            yield i, batches
        else:
            break

class GridWorld_pendulum(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    NOTE: in original Lyapunov NN, the grid is defined in a normalized 
          fashion (i.e. [-1, 1] for each dimension)
    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld_pendulum, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int16, copy=False)
        self.state_dim = len(self.limits)
        # print('self.state_dim: ', self.state_dim)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)

        self.ndim = len(self.limits)
        self._all_points = None

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            # my own implementation
            mesh = np.stack(np.meshgrid(*self.discrete_points),-1).reshape(-1,self.state_dim)
            self._all_points = mesh.astype(config.np_dtype)
            if self.all_points.shape[1] == 2:
                # swap the first two columns
                self._all_points[:,[0,1]] = self._all_points[:,[1,0]]

            # original implementation
            # mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            # points = np.column_stack(col.ravel() for col in mesh)
            # each row of the mesh is a point in the stat space
            # self._all_points = points.astype(config.np_dtype)

        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)
