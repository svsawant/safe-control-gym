


######################## define Lyapunov NN ########################
# initialize Lyapunov NN
import torch
import numpy as np
# from lyapnov import LyapunovNN/

class LyapunovNN(torch.nn.Module):
    # def __init__(self, dim_input, layer_dims, activations):
    def __init__(self, input_dim, layer_dims, activations, eps=1e-6):
        super(LyapunovNN, self).__init__()
        # network layers
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.eps = eps
        self.layers = torch.nn.ModuleList()
        self.kernel = []

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

        # build the nn structure
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.layers.append(\
                        torch.nn.Linear(layer_input_dim, self.hidden_dims[i], bias=False))
            W = self.layers[-1].weight
            kernel = torch.matmul(W.T, W) + self.eps * torch.eye(W.shape[1])
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                self.layers.append(torch.nn.Linear(layer_input_dim, dim_diff, bias=False))
                # print(kernel.shape, self.layers[-1].weight.shape)
                kernel = torch.cat((kernel,self.layers[-1].weight), dim=0)
            self.kernel.append(kernel)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        for i in range(self.num_layers):
            layer_output = torch.matmul(self.kernel[i], x)
            x = self.activations[i](layer_output)
        values = torch.sum(torch.square(x), dim=-1)
        return values

    def print_params(self):
        offset = 0
        # get nn parameters
        params = []
        for _, param in self.named_parameters():
            params.append(param.data)

        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            print('Layer {}:'.format(i))
            print('dim_diff: ', dim_diff)

            print('Layer weights {}:'.format(i))
            W0 = params[offset + i]
            print('W0:\n{}'.format(W0))
            if dim_diff > 0:
                W1 = params[offset + 1 + i]
                print('W1:\n{}'.format(W1))
            else:
                offset += 1
            kernel = W0.T @ (W0) + self.eps * np.eye(W0.shape[1])
            eigvals, _ = np.linalg.eig(kernel)
            print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')

state_dim = 2

layer_dim = [64, 64, 64]
# layer_dim = [128, 128, 128]
activations = [torch.nn.Tanh(), torch.nn.Tanh(), torch.nn.Tanh()]
nn = LyapunovNN(state_dim, layer_dim, activations)
print('nn: ', nn)
print(nn.hidden_dims)
print('nn.kernel[0]', nn.kernel[0].shape)
print('nn.kernel[1]', nn.kernel[1].shape)
print('nn.kernel[2]', nn.kernel[2].shape)

# for name, param in nn.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# forward some random states
x = torch.randn(state_dim,)
print('x: ', x)
print('nn(x): ', nn(x))

# nn.print_params()

# for name, param in nn.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)