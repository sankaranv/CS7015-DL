'''
def gradient_descent(lr):
    i = 0
    max_iter = 1000
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_a, grad_W, grad_b, grad_h = backward(y_true, y_pred)
        for key in W:
            self.W[key] = self.W[key] - lr*grad_W[key]
        for key in b:
            self.b[key] = self.b[key] - lr*grad_b[key]

def momentum_gradient_descent(lr, momentum):
    i = 0
    max_iter = 1000
    update_W, update_b, prev_update_W, prev_update_b = {}, {}, {}, {}
    for key in self.W:
        prev_update_W[key] = np.zeros(shape=self.W[key].shape)
    for key in self.b:
        prev_update_b[key] = np.zeros(shape=self.b[key].shape)
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_a, grad_W, grad_b, grad_h = backward(y_true, y_pred)
        for key in self.W:
            update_W[key] = momentum * prev_update_W[key] + lr * grad_W[key]
            self.W[key] = self.W[key] - update_W[key]
            prev_update_W[key] = update_W[key]
        for key in self.b:
            update_b[key] = momentum * prev_update_b[key] + lr * grad_b[key]
            self.b[key] = self.b[key] - update_b[key]
            prev_update_b[key] = update_b[key]
'''
###############################################################################

def gradient_descent(lr):
    i = 0
    max_iter = 1000
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_params = backward(y_true, y_pred)
        params = params - lr*grad_params

def momentum_gradient_descent(lr, momentum):
    i = 0
    max_iter = 1000
    prev_update = np.zeros(shape=params.shape)
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_params = backward(y_true, y_pred)
        update = momentum * prev_update + lr * grad_params
        params = params - update
        prev_update = update

def nesterov_accelerated_gradient_descent(lr, momentum):
    i = 0
    max_iter = 1000
    prev_update = np.zeros(shape=params.shape)
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_params = backward(y_true, y_pred)
        update = momentum * prev_update + lr * grad_params
        params = params - update
        prev_update = update

def adam_optimizer(beta_1=0.9, beta_2=0.999, lr, epsilon=1e-9):
    i = 0
    max_iter = 1000
    prev_momentum = np.zeros(shape=params.shape)
    prev_velocity = np.zeros(shape=params.shape)
    while (i <= max_iter):
        y_pred, loss = forward(x, y)
        grad_params = backward(y_true, y_pred)
        momentum = beta1*prev_momentum + (1-beta1)*grad_params
        velocity = beta2*prev_velocity + (1-beta2)*np.square(grad_params)
        momentum = momentum / (1 - np.power(beta1,(i+1)))
        velocity = velocity / (1 - np.power(beta2,(i+1)))
        params = params - np.multiply((lr / np.power(velocity+epsilon,0.5)), momentum)
