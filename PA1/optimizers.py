import numpy as np

def gradient_descent(network, x, y, lr):
    y_pred, loss = network.forward(x, y)
    network.backward(y, y_pred)
    network.theta[:] = network.theta - lr * network.grad_theta
    return loss

def momentum_gradient_descent(network, x, y, lr, momentum):
    y_pred, loss = network.forward(x, y)
    network.backward(y, y_pred)
    update = momentum * network.prev_velocity + lr * network.grad_theta
    network.theta[:] = network.theta - update
    network.prev_velocity = update
    return loss

def adam_optimizer(network, x, y, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-9):
    y_pred, loss = network.forward(x, y)
    network.backward(y, y_pred)
    momentum = beta1*prev_momentum + (1-beta1)*network.grad_theta
    velocity = beta2*prev_velocity + (1-beta2)*np.square(network.grad_theta)
    momentum = momentum / (1 - np.power(beta1,(i+1)))
    velocity = velocity / (1 - np.power(beta2,(i+1)))
    network.theta[:] = network.theta - np.multiply((lr / np.power(velocity+epsilon,0.5)), momentum)
    network.prev_velocity = velocity
    network.prev_momentum = momentum
    return loss
