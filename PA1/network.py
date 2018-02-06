import numpy as np

def activation_function(x, activation=activation):
    if (activation == 'sigmoid'):
        return 1/(1+np.exp(-x))
    elif (activation == 'tanh'):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    else:
        return x

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
    def __init__(self, activation = 'sigmoid', loss = 'ce'):
        I, O = 784, 10
        self.W = {}
        self.b = {}

        self.W['W1'] = np.random.uniform(size = (I, sizes[0]))
        self.b['b1'] = np.random.uniform(size = (1, sizes[0]))
        for i in range(2, L + 1):
            self.W['W{}'.format(i)] = np.random.uniform(size = (sizes[i - 1], sizes[i]))
            self.b['b{}'.format(i)] = np.random.uniform(size = (1, sizes[i]))
        self.W['Wo'] = np.random.uniform(size = (sizes[L-1], O))
        self.b['bo'] = np.random.uniform(size = (1, O))

        self.activation = activation
        self.loss = loss

    def forward(self, x, y):
        h, a = {}, {}
        h['h0'] = x
        for i in range (1, L + 1):
            a['a{}'.format(i)] = self.b['b{}'.format(i)] + np.matmul( h['h{}'.format(i-1)], self.W['W{}'.format(i)])
            h['h{}'.format(i)] = activation_function(a['a{}'.format(i)], self.activation)
        a['aO'] = self.b['bO'] + np.matmul(h['hL'], self.W['WO'])
        y_pred = output_function(a['aO'])
        loss = loss(y, y_pred, self.loss)
        return y_pred, loss

    def backward():
        pass
