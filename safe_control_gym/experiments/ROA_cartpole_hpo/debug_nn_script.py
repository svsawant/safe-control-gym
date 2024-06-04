import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchviz import make_dot

# fix random seed
np.random.seed(0)
torch.manual_seed(0)


class LyapunovNN_hardcoded(torch.nn.Module):
    # def __init__(self, dim_input, layer_dims, activations):
    def __init__(self, input_dim, layer_dims, activations, eps=1e-6):
        super(LyapunovNN_hardcoded, self).__init__()
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
   
        # build the nn structure (hardcoded)
        
        # layer as parameters
        self.layers.append(torch.nn.Linear(2, 2, bias=False))
        self.layers.append(torch.nn.Linear(2, 2, bias=False))
        W0 = self.layers[0].weight
        W1 = self.layers[1].weight
        ############################### possible bug here ########################
        # weight0 = W0
        # weight1 = W1
        # if use clone, weights are updated but the kernel is not updated
        weight0 = W0.clone()
        weight1 = W1.clone()
        ###########################################################################
        # kernels follow the equation in Remark 1 (last page) in the paper
        self.kernel = [torch.matmul(weight0.T, weight0) + self.eps * torch.eye(W0.shape[1]),
                       torch.matmul(weight1.T, weight1) + self.eps * torch.eye(W1.shape[1])]

    def update_kernel(self):
        # update the kernel
        W0 = self.layers[0].weight
        W1 = self.layers[1].weight
        self.kernel = [torch.matmul(W0.T, W0) + self.eps * torch.eye(W0.shape[1]),
                       torch.matmul(W1.T, W1) + self.eps * torch.eye(W1.shape[1])]

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        # kernel => activation => kernel => activation 
        # => sum of squares => positive definite output
        x = self.activations[0](torch.matmul(self.kernel[0], x))
        x = self.activations[1](torch.matmul(self.kernel[1], x))
        x = torch.sum(torch.square(x))
        return x
    
    ###########################################################################

def get_points_on_ellipsoid(P, c , n_points=100, radius=1.0):
    # Determine the inverse of the square root of the matrix P
    P_sqrt_inv = np.linalg.inv(np.sqrt(P))
    # Create a circle in the 2d space
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.vstack((np.cos(theta), np.sin(theta)))
    # Scale the circle with the inverse of the square root of P
    ellipse = P_sqrt_inv @ circle
    # Translate the ellipse to the center c
    ellipse = ellipse + c.reshape(-1, 1)
    return ellipse



def run():
    '''
    Goal: train a neural network to approximate the following Lyapunov function
    '''
    P = np.array([1, 0.5, 0.5, 1]).reshape(2, 2)
    def ellipse(x):
        return x.T @ P @ x 
    # get points on the ellipse
    num_points = 1000
    points = get_points_on_ellipsoid(P, c=np.zeros(2), n_points=num_points)
    # plot the ellipse
    # plt.plot(points[0, :], points[1, :])


    ##################### define the neural network #####################
    nn = LyapunovNN_hardcoded(2, [2, 2], [torch.nn.Tanh(), torch.nn.Tanh()])
    # inspect positive definiteness of each layer
    # printed eigenvalues should be positive
    print(nn.layers[0].weight)
    print(nn.kernel[0])
    eigvals, _ = np.linalg.eig(nn.kernel[0].detach().numpy())
    print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')

    print(nn.layers[1].weight)
    print(nn.kernel[1])
    eigvals, _ = np.linalg.eig(nn.kernel[1].detach().numpy())
    print('Eigenvalues of (W1.T*W1 + eps*I):', eigvals, '\n')
    print('-------------------------------------')
    ############################# create nn graph ########################
    # create dummy test input
    x_test = [torch.tensor([0., 0.]), torch.tensor([0.5, 0.5]) ,torch.tensor([1., 1.])]
    # forward pass
    y_output = [nn(x) for x in x_test]
    y_test = [ellipse(x.detach().numpy()) for x in x_test]
    # make graph
    dir_path = os.path.dirname(os.path.realpath(__file__))
    graph_file_name = 'graph'
    graph_file_path = os.path.join(dir_path, graph_file_name)
    make_dot(y_output[0], params=dict(list(nn.named_parameters())))\
                                .render(graph_file_path, format="png")

    ############################# hyperparameters ########################
    # create optimizer
    learning_rate = 0.1
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)
    training_epochs = 100
    num_data_in_batch = 100
    print_loss_every = 50

    ############################# train the nn ########################
    torch.autograd.set_detect_anomaly(True)
    nn.train(True)
    # train the neural network
    for i in range(training_epochs):
        # sample 100 state from the data
        idx = np.random.choice(num_points, num_data_in_batch)
        input = points[:, idx].copy()
        # forward pass
        # create empty tensor to store the output
        output = torch.empty(num_data_in_batch)
        # loop through the input
        for i in range(num_data_in_batch):
            # get the output of the neural network
            output[i] = nn(input[:, i])
        # compute loss
        loss = torch.mean(output)

        # zero gradients
        optimizer.zero_grad()
        # backward pass
        ''' 
        possible bug here 
        '''
        # loss.backward()
        loss.backward(retain_graph=True)

        # print weights before update
        print('-------------------------------------')
        print('weights before update')
        print('nn.layers[0].weight\n', nn.layers[0].weight)
        print('nn.layers[1].weight\n', nn.layers[1].weight)
        # print the gradient
        print('-------------------------------------')
        print('gradient')
        print('nn.layers[0].weight.grad\n', nn.layers[0].weight.grad)
        print('nn.layers[1].weight.grad\n', nn.layers[1].weight.grad)

        # test states before update
        y_output_before = [nn(x) for x in x_test]

        # update parameters
        print('--------------nn updates----------------')
        optimizer.step()
        nn.update_kernel()
        # print weights after update
        print('-------------------------------------')
        print('weights after update')
        print('nn.layers[0].weight\n', nn.layers[0].weight)
        print('nn.layers[1].weight\n', nn.layers[1].weight)

        # test states after update
        y_output_after = [nn(x) for x in x_test]
        # print output
        print('y_output\n', y_output_before)
        print('y_output\n', y_output_after)

        # print loss every 10 iterations
        if i % print_loss_every == 0:
            print('loss: ', loss.item())
            print('-------------------------------------')

    ############################# plot the nn ########################
    # plot the points after passing through the nn
    nn.train(False)
    plt.figure()
    plt.plot(points[0, :], points[1, :], 'b.')
    y_output = [nn(x) for x in points]
    
    plt.show()

if __name__ == '__main__':
    run()  
    plt.show()