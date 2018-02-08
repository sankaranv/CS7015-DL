import numpy as np

def activation_function(x, activation=activation):
    if (activation == 'sigmoid'):
        return 1/(1+np.exp(-x))
    elif (activation == 'tanh'):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    else:
        return x

def grad_activation_function(x, activation=activation):
    a = activation_function(x, activation=activation)
    if (activation == 'sigmoid'):
        return a*(1-a)
    elif (activation == 'tanh'):
        return 1-(a**2)
    else:
        return 1.

def output_function(x, activation='softmax'):
    if (activation == 'softmax'):
        x = np.exp(x - np.max(x))  # Normalization for numerical stability, from CS231n notes
        return x/np.sum(x, axis=0)
    if (activation == 'sigmoid'):
        return 1/(1+np.exp(-x))
    else:
        return x

def loss(y_true, y_pred, loss=loss):
    batch_size = y_true.shape[0]
    if loss == 'sq':
        return (1./(2*batch_size)) * np.sum((y_true - y_pred)**2)
    if loss == 'ce':
        return (-1./batch_size) * np.sum*(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

class Network:

    def __init__(self, activation = 'sigmoid', loss = 'ce', sizes):
        # L hidden layers, layer 0 is input, layer (L+1) is output
        num_input, num_output = 784, 10
        self.W = {}
        self.b = {}
        self.L = len(num_hidden)
        self.W['W1'] = np.random.uniform(size = (num_input, sizes[0]))
        self.b['b1'] = np.random.uniform(size = (1, sizes[0]))
        for i in range(2, L + 1):
            self.W['W{}'.format(i)] = np.random.uniform(size = (sizes[i - 1], sizes[i]))
            self.b['b{}'.format(i)] = np.random.uniform(size = (1, sizes[i]))
        self.W['W{}'.format(L + 1)] = np.random.uniform(size = (sizes[L-1], num_output))
        self.b['b{}'.format(L + 1)] = np.random.uniform(size = (1, num_output))
        self.activation = activation
        self.loss = loss

    def forward(self, x, y):
        # a(i) = b(i) + W(i)*h(i-1)
        # h(i) = g(i-1)
        self.h, self.a = {}, {}
        h['h0'] = x
        for i in range (1, L):
            a['a{}'.format(i)] = self.b['b{}'.format(i)] + np.matmul(h['h{}'.format(i-1)], self.W['W{}'.format(i)])
            h['h{}'.format(i)] = activation_function(a['a{}'.format(i)], self.activation)
        a['a{}'.format(L)] = self.b['b{}'.format(L)] + np.matmul(h['h{}'.format(L-1)], self.W['W{}'.format(L)])
        y_pred = output_function(a['a{}'.format(L)])
        loss = loss(y, y_pred, self.loss)
        return y_pred, loss

    def backward(y_true, y_pred):
        grad_a, grad_W, grad_b, grad_h = {}
        # Compute output gradient
        e_y = np.zeros(shape=y_pred.shape)
        e_y[np.argmax(y_true)]=1
        grad_a['a{}'.format(L)] = -(e_y - y_pred)
        for k in range (L, 0, -1):
            # Gradients wrt parameters
            grad_W['W{}'.format(k)] = np.matmul(grad_a['a{}'.format(k)], self.h['h{}'.format(k-1)].T)
            grad_b['b{}'.format(k)] = grad_a['a{}'.format(k)]
            # Gradients wrt prev layer
            grad_h['h{}'.format(k-1)] = mp.matmul(self.W['W{}'.format(k)].T, grad_a['a{}'.format(k)])
            # Gradients wrt prev preactivation
            grad_a['a{}'.format(k-1)] = np.multiply(grad_h['h{}'.format(k-1)], grad_activation_function(a['a{}'.format(k-1)], self.activation))
        return grad_a, grad_W, grad_b, grad_h

    def train():
