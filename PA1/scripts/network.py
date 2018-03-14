import numpy as np
from helpers import activation_function, output_function, loss_function

class Network:

    def __init__(self, num_hidden, sizes, activation_choice = 'softmax', output_choice = 'softmax', loss_choice = 'ce'):
        # L hidden layers, layer 0 is input, layer (L+1) is output
        self.sizes = sizes
        sizes = [784] + sizes + [10]
        self.L = num_hidden
        self.output_shape = 10
        # Parameter map from theta to Ws and bs
        self.param_map = {}
        start, end = 0, 0
        for i in range(1, self.L + 2):
            end = start + sizes[i - 1] * sizes[i]
            self.param_map['W{}'.format(i)] = (start, end)
            start = end
            end = start + sizes[i]
            self.param_map['b{}'.format(i)] = (start, end)
            start = end
        num_params = end
        # Parameter vector - theta
        self.theta = np.random.uniform(-1.0, 1.0, num_params)
        # Gradient vector - theta
        self.grad_theta = np.zeros_like(self.theta)
        # Map theta (grad_theta) to params (grad_params)
        self.params = {}
        self.grad_params = {}
        for i in range(1, self.L + 2):
            weight = 'W{}'.format(i)
            start, end = self.param_map[weight]
            self.params[weight] = self.theta[start : end].reshape((sizes[i], sizes[i - 1]))
            self.grad_params[weight] = self.grad_theta[start : end].reshape((sizes[i], sizes[i - 1]))
            bias = 'b{}'.format(i)
            start, end = self.param_map[bias]
            self.params[bias] = self.theta[start : end].reshape((sizes[i], 1))
            self.grad_params[bias] = self.grad_theta[start : end].reshape((sizes[i], 1))

        self.activation_choice = activation_choice
        self.output_choice = output_choice
        self.loss_choice = loss_choice

    # x is of shape (input_size, batch_size), y is of shape (batch_size)
    def forward(self, x, y):
        # a(i) = b(i) + W(i)*h(i-1)
        # h(i) = g(i-1)
        self.activations = {}
        self.activations['h0'] = x
        self.batch_size = x.shape[1]
        for i in range (1, self.L + 1):
            self.activations['a{}'.format(i)] = self.params['b{}'.format(i)] + np.matmul(self.params['W{}'.format(i)], self.activations['h{}'.format(i-1)])
            self.activations['h{}'.format(i)] = activation_function(self.activations['a{}'.format(i)], self.activation_choice)

        self.activations['a{}'.format(self.L + 1)] = self.params['b{}'.format(self.L + 1)] + np.matmul(self.params['W{}'.format(self.L+1)], self.activations['h{}'.format(self.L)])
        y_pred = output_function(self.activations['a{}'.format(self.L + 1)], self.output_choice)
        loss = loss_function(y, y_pred, self.loss_choice)
        return y_pred, loss

    def backward(self, y_true, y_pred):
        grad_activations = {}
        # Compute output gradient
        e_y = np.zeros_like(y_pred)
        e_y[y_true, range(self.batch_size)] = 1
        if self.loss_choice == 'ce':
            grad_activations['a{}'.format(self.L + 1)] = -(e_y - y_pred)
        elif self.loss_choice == 'sq':
            grad_activations['a{}'.format(self.L + 1)] = -(e_y - y_pred) * y_pred * (1 - y_pred)
        for k in range (self.L + 1, 0, -1):
            # Gradients wrt parameters
            self.grad_params['W{}'.format(k)][:, :] = (1.0 / self.batch_size) * np.matmul(grad_activations['a{}'.format(k)], self.activations['h{}'.format(k-1)].T)
            self.grad_params['b{}'.format(k)][:, :] = (1.0 / self.batch_size) * np.sum(grad_activations['a{}'.format(k)], axis = 1, keepdims = True)
            # Do not compute gradients with respect to the inputs
            if k == 1:
                break
            # Gradients wrt prev layer
            grad_activations['h{}'.format(k-1)] = np.matmul(self.params['W{}'.format(k)].T, grad_activations['a{}'.format(k)])
            # Gradients wrt prev preactivation
            if self.activation_choice == 'sigmoid':
                grad_activation_ = np.multiply(self.activations['h{}'.format(k - 1)], 1 - self.activations['h{}'.format(k - 1)])
            elif self.activation_choice == 'tanh':
                grad_activation_ = 1 - (self.activations['h{}'.format(k - 1)]) ** 2
            elif self.activation_choice == 'relu':
                grad_activation_ = np.zeros_like(self.activations['a{}'.format(k - 1)])
                grad_activation_[np.where(self.activations['a{}'.format(k - 1)] > 0)] = 1.0
            grad_activations['a{}'.format(k-1)] = np.multiply(grad_activations['h{}'.format(k-1)], grad_activation_)

    def performance(self, y_true, y_pred):
        y_pred = y_pred.argmax(axis = 0)
        return float(np.sum(y_pred != y_true)) /y_pred.shape[0] * 100

    def predict(self, x):
        y_pred, _ = self.forward(x, np.ones((x.shape[1]), dtype = np.int16))
        return y_pred.argmax(axis = 0)

    def save(self, path):
        np.save(path, self.theta)

    def load(self, theta = None, path = None):
        if path != None:
            theta = np.load(path)
        self.theta[:] = theta

