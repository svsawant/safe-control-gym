import casadi as cs
import numpy as np


def euler_discrete(f, n, m, dt):
    '''Euler discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    '''
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    x_next = X + dt * f(X, U)
    eu_dyn = cs.Function('eu_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return eu_dyn

# Adam Optimizer (use the class from the previous response)


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Constructor for the AdamOptimizer class.

        Parameters
        ----------
        learning_rate : float
            Learning rate for the optimizer.
        beta1 : float
            Exponential decay rate for the first moment estimates.
        beta2 : float
            Exponential decay rate for the second moment estimates.
        epsilon : float
            Small value to prevent division by zero.

        Returns
        -------
        None.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize_moments(self, params):
        """
        Initializes the first and second moment estimates.

        Parameters
        ----------
        params : dict
            Dictionary containing the model parameters.

        Returns
        -------
        None.
        """
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update_params(self, params, grads):
        """
        Updates the model parameters using the Adam optimizer.

        Parameters
        ----------
        params : dict
            Dictionary containing the model parameters.
        grads : dict
            Dictionary containing the gradients for each parameter.

        Returns
        -------
        updated_params : dict
            Dictionary containing the updated model parameters.
        """
        if self.m is None or self.v is None:
            self.initialize_moments(params)

        self.t += 1
        updated_params = {}

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])

            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)

            updated_params[key] = params[key] - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return updated_params
