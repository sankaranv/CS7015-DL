# System libraries
import sys
import os
import argparse
import logging
import numpy as np

# Custom imports
from helpers import setup_logger
from network import Network
from optimizers import gradient_descent
from data import loadData

# Parse Arguments

parser = argparse.ArgumentParser(description='Train the MLP')
parser.add_argument('--lr', type=float,
                    help='initial learning rate for gradient descent')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--num_hidden', type=int, default=1,
                    help='number of hidden layers')
parser.add_argument('--sizes', type=lambda x: x.split(','),
                    help='sizes of hidden layers')
parser.add_argument('--activation', default='sigmoid',
                    help='activation function (tanh or sigmoid)')
parser.add_argument('--loss', default='sq',
                    help='loss function (sq or ce)')
parser.add_argument('--opt', default='adam',
                    help='loss function (gd, momentum, nag, or adam)')
parser.add_argument('--batch_size', type=int,
                    help='batch size (multiples of 5)')
parser.add_argument('--anneal', type=bool, default=False,
                    help='halve the learning rate if at any epoch the \
                    validation loss decreases and then restart that epoch')
parser.add_argument('--save_dir', default='./models',
                    help='save dir for model')
parser.add_argument('--expt_dir', default='./logs',
                    help='save dir for experiment logs')
parser.add_argument('--train', default='./data',
                    help='path to training set')
parser.add_argument('--val', default='./data',
                    help='path to validation set')
parser.add_argument('--test', default='./data',
                    help='path to test set')
args = parser.parse_args()

if (args.lr is None):
    print "Provide learning rate"
    sys.exit(1)
else:
    lr = args.lr

if ((args.activation != 'sigmoid') and (args.activation != 'tanh')):
    print "Activation is sigmoid/tanh"
    sys.exit(1)
else:
    activation = args.activation

if ((args.loss != 'sq') and (args.loss != 'ce')):
    print "Loss function is sq/ce"
    sys.exit(1)
else:
    loss = args.loss

if ((args.opt != 'gd') and (args.opt != 'momentum') and (args.opt != 'nag') and (args.opt != 'adam')):
    print "Optimizer is gd/momentum/nag/adam"
    sys.exit(1)
else:
    opt = args.opt

if (args.batch_size is None or (args.batch_size % 5 != 0 and args.batch_size != 1)):
    print "Batch size is 1 or a multiple of 5"
    sys.exit(1)
else:
    batch_size = args.batch_size

if (len(args.sizes) != args.num_hidden):
    print "Sizes don't match number of hidden layers"
    sys.exit(1)
else:
    sizes = map(int, args.sizes)
    num_hidden = args.num_hidden
    L = args.num_hidden

momentum = args.momentum
anneal = True
# Paths
train_path, valid_path, test_path = args.train, args.val, args.test
model_path = args.save_dir
logs_path = args.expt_dir

# Logging
train_log = setup_logger('train-log', os.path.join(logs_path, 'train.log'))
valid_log = setup_logger('valid-log', os.path.join(logs_path, 'valid.log'))
# Load data
data = loadData(train_path, valid_path, test_path)
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                     data['valid']['X'], data['valid']['Y'],\
                                                     data['test']['X'], data['test']['Y'], 

# Initialize network
np.random.seed(1234)
network = Network(num_hidden, sizes, activation_choice = activation, output_choice = 'softmax', loss_choice = loss)
model_name = '{}-{}-{}-{}-{}-{}.npy'.format(num_hidden, ','.join([str(word) for word in sizes]), activation, 'softmax', loss, lr)
# Train
num_epochs = 100
num_batches = int(float(train_X.shape[0]) / batch_size)
steps = 0
latency = 25
loss_history = [np.inf]
for epoch in range(num_epochs):
    steps = 0
    for batch in range(num_batches):
        start, end = batch * batch_size, (batch + 1) * batch_size
        x, y = train_X[:, range(start, end)], train_Y[range(start, end)]
        #gradient_descent(network, x, y, lr)
        grad_check(network, x, y, lr)
        steps += batch_size
        if steps % 100 == 0 and steps != 0:
            y_pred, loss = network.forward(train_X, train_Y)
            error = network.performance(train_Y, y_pred)
            train_log.info('Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}'.format(epoch, steps, loss, error, lr))
            y_pred, loss = network.forward(valid_X, valid_Y)
            error = network.performance(valid_Y, y_pred)
            valid_log.info('Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}'.format(epoch, steps, loss, error, lr))
            if loss < min(loss_history):
                network.save(os.path.join(model_path, model_name))    
            loss_history.append(loss)
            latency -= 1
            print anneal, len(loss_history) - np.argmin(loss_history)
            if anneal == True and len(loss_history) - np.argmin(loss_history) > 10 and latency < 0:
                network.load(os.path.join(model_path, model_name))
                lr /= 2
                latency = 25
            if lr <= 1e-4:
                print 'Training ended'
                exit()

