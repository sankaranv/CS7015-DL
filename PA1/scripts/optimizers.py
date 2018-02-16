import numpy as np
from network import Network

class Optimizers:
    def __init__(self, num_params, choice = 'gd', lr = 0.001, momentum = 0.9):
        self.opts = {'gd' : self.gradient_descent, 'momentum' : self.momentum_gradient_descent,\
                     'adam' : self.adam_optimizer, 'grad_check' : self.grad_check}
        self.lr = lr
        # To help with momentum based optimizers
        self.momentum = momentum
        self.prev_square = np.zeros_like(num_params)
        self.prev_update = np.zeros_like(num_params)
        self.iter = 1

    def gradient_descent(self, network, x, y):
        y_pred, loss = network.forward(x, y)
        network.backward(y, y_pred)
        network.theta[:] = network.theta - self.lr * network.grad_theta
        return loss

    def momentum_gradient_descent(self, network, x, y):
        y_pred, loss = network.forward(x, y)
        network.backward(y, y_pred)
        update = self.momentum * self.prev_velocity + self.lr * network.grad_theta
        network.theta[:] = network.theta - update
        self.prev_update = update
        return loss

    def adam_optimizer(self, network, x, y):
        beta_1 = 0.9 
        beta_2 = 0.999
        epsilon = 1e-9
        y_pred, loss = network.forward(x, y)
        network.backward(y, y_pred)
        update = beta_1 * self.prev_update + (1-beta_1) * network.grad_theta
        square = beta_2 * self.prev_square + (1-beta_2) * np.square(network.grad_theta)
        update = update / (1 - np.power(beta_1,(self.iter)))
        square = square / (1 - np.power(beta_2,(self.iter)))
        network.theta[:] = network.theta - self.lr * (update / (np.sqrt(square) + epsilon))
        self.prev_update = update
        self.prev_square = square
        self.iter += 1
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

