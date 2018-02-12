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

def grad_check(network, x, y, lr, epsilon=1e-7):
    y_pred, loss = network.forward(x, y)
    print "Loss " + str(loss)
    network.backward(y, y_pred)
    print network.theta.shape
    testnet_plus = Network(network.L, network.sizes, activation_choice = network.activation_choice, output_choice = network.output_choice, loss_choice = network.loss_choice)
    testnet_minus = Network(network.L, network.sizes, activation_choice = network.activation_choice, output_choice = network.output_choice, loss_choice = network.loss_choice)
    loss_plus = np.zeros(shape=network.theta.shape)
    loss_minus = np.zeros(shape=network.theta.shape)
    grad_approx = np.zeros(shape=network.theta.shape)
    for i in range(len(network.theta)):
        print "Param " + str(i) + "    ",
        testnet_plus.theta = np.copy(network.theta)
        testnet_plus.theta[i] = testnet_plus.theta[i] + epsilon
        _, loss_plus[i] = testnet_plus.forward(x, y)
        print "+ " + str(loss_plus[i]) + "    ",
        testnet_minus.theta = np.copy(network.theta)
        testnet_minus.theta[i] = testnet_minus.theta[i] - epsilon
        _, loss_minus[i] = testnet_plus.forward(x, y)
        print "- " + str(loss_minus[i]) + "    ",
        grad_approx[i] = (loss_plus[i] - loss_minus[i])/ (2*epsilon)
        print "Approx " + str(grad_approx[i]) + "    "
    difference = np.linalg.norm(network.grad_theta - grad_approx)/(np.linalg.norm(network.grad_theta) + np.linalg.norm(grad_approx))
    if difference > epsilon :
        print "There is a problem with the gradients! Difference = " + str(difference)
    else:
        print "Code seems to be fine! Difference = " + str(difference)
    network.theta[:] = network.theta - lr * network.grad_theta
    return loss
