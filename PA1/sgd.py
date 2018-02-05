a = {}
h = {}

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

def forward(x, W, b):
    h['h0'] = x
    for i in range (1, L):
        a['a{0}'.format(i)] = b['b{0}'.format(i)] + np.dot(W['W{0}'.format(i)], h['h{0}'.format(i-1)])
        h['h{0}'.format(i)] = activation_function(a['a{0}'.format(i)])
    a['a{0}'.format(L)] = b['b{0}'.format(L)] + np.dot(W['W{0}'.format(L)], h['h{0}'.format(L-1)])
    y_pred = output_function(a['a{0}'.format(L)])
    return y_pred

def loss(y_true, y_pred, batch_size, loss=loss):
    if (loss='sq'):
        return (1./(2*batch_size)) * np.sum((y_true - y_pred)**2)
    if (loss='ce'):
        return (-1./batch_size) * np.sum*(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

grad_a = {}
grad_h = {}
grad_W = {}
grad_b = {}

def backward(y_true, y_pred, a, h):
    # One hot vector
    e = np.zeros(shape=y_true.shape)
    e[np.argmax(y_true)] = 1
    # Gradient wrt output
    grad_a['a{0}'.format(L)] = -(e - y_pred)
