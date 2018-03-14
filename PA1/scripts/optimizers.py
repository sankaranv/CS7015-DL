import numpy as np
from network import Network

class Optimizers:
    def __init__(self, num_params, choice = 'gd', lr = 0.001, momentum = 0.9):
        self.opts = {'gd' : self.gradient_descent, 'momentum' : self.momentum_gradient_descent,\
                     'nag' : self.nesterov_accelerated_gradient_descent,\
                     'adam' : self.adam_optimizer, 'grad_check' : self.grad_check}
        self.lr = lr
        # To help with momentum based optimizers
        self.momentum = momentum
        self.grad_square_history = np.zeros_like(num_params)
        self.grad_history = np.zeros_like(num_params)
        self.iter = 0

    def gradient_descent(self, network, x, y):
        y_pred, loss = network.forward(x, y)
        network.backward(y, y_pred)
        network.theta[:] = network.theta - self.lr * network.grad_theta
        return loss

    def momentum_gradient_descent(self, network, x, y):
        y_pred, loss = network.forward(x, y)
        network.backward(y, y_pred)
        self.grad_history = self.momentum * self.grad_history + self.lr * network.grad_theta
        network.theta[:] = network.theta - self.grad_history
        return loss

    def nesterov_accelerated_gradient_descent(self, network, x, y):
        y_pred, loss = network.forward(x, y)
        look_ahead = Network(network.L, network.sizes, network.activation_choice, network.output_choice, network.loss_choice)
        look_ahead.load(theta = network.theta - self.momentum * self.grad_history)
        look_ahead.forward(x, y)
        look_ahead.backward(y, y_pred)
        self.grad_history = self.momentum * self.grad_history + self.lr * look_ahead.grad_theta
        network.theta[:] = network.theta - self.grad_history
        return loss
        
    def adam_optimizer(self, network, x, y):
        beta_1 = 0.9 
        beta_2 = 0.999
        epsilon = 1e-8
        self.iter += 1
        y_pred, loss = network.forward(x, y)
        network.backward(y, y_pred)
        self.grad_history = beta_1 * self.grad_history + (1 - beta_1) * network.grad_theta
        self.grad_square_history = beta_2 * self.grad_square_history + (1 - beta_2) * np.square(network.grad_theta)
        update = self.grad_history / (1 - beta_1 ** self.iter)
        square = self.grad_square_history / (1 - beta_2 ** self.iter)
        network.theta[:] = network.theta - self.lr * (update / (np.sqrt(square) + epsilon))
        return loss

    def grad_check(self, network, x, y):
        epsilon = 1e-7
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
            #print "Param " + str(i) + "    ",
            testnet_plus.load(theta = network.theta)
            testnet_plus.theta[i] = testnet_plus.theta[i] + epsilon
            _, loss_plus[i] = testnet_plus.forward(x, y)
            #print "+ " + str(loss_plus[i]) + "    ",
            testnet_minus.load(theta = network.theta)
            testnet_minus.theta[i] = testnet_minus.theta[i] - epsilon
            _, loss_minus[i] = testnet_minus.forward(x, y)
            #print "- " + str(loss_minus[i]) + "    ",
            grad_approx[i] = (loss_plus[i] - loss_minus[i])/ (2*epsilon)
            #print "Approx " + str(grad_approx[i]) + "    "
        difference = np.linalg.norm(network.grad_theta - grad_approx)/(np.linalg.norm(network.grad_theta) + np.linalg.norm(grad_approx))
        #print np.linalg.norm(network.grad_theta - grad_approx)
        if difference > epsilon * 100 :
            print "There is a problem with the gradients! Difference = " + str(difference)
            exit()
        else:
            print "Code seems to be fine! Difference = " + str(difference)
        network.theta[:] = network.theta - self.lr * network.grad_theta
        return loss

