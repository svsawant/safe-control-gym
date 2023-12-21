

import numpy as np
import torch
# Add the configuration settings



class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = torch.float64

        # Batch size for stability verification
        self.gp_batch_size = 10000

    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        return np.float64

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

    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int16, copy=False)

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
            mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            points = np.column_stack(col.ravel() for col in mesh)
            self._all_points = points.astype(config.np_dtype)

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