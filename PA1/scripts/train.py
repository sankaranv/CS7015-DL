# System libraries
import sys
import os
import argparse
import logging
import numpy as np

# Custom imports
from helpers import setup_logger
from network import Network
from optimizers import Optimizers
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
parser.add_argument('--anneal', default="true",
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
parser.add_argument('--pretrain', default = None,
                    help='path to pretrained model')
args = parser.parse_args()

if (args.lr is None):
    print "Provide learning rate"
    sys.exit(1)
else:
    lr = args.lr

if ((args.activation != 'sigmoid') and (args.activation != 'tanh') and (args.activation != 'relu')):
    print "Activation is sigmoid/tanh"
    sys.exit(1)
else:
    activation = args.activation

if ((args.loss != 'sq') and (args.loss != 'ce')):
    print "Loss function is sq/ce"
    sys.exit(1)
else:
    loss = args.loss
    # Output function
    output_choice = 'softmax' if loss == 'ce' else 'sigmoid'

if ((args.opt != 'gd') and (args.opt != 'momentum') and (args.opt != 'nag') and (args.opt != 'adam') and (args.opt != 'grad_check')):
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
if args.anneal.lower() == "true":
    anneal = True
else:
    anneal = False

momentum = args.momentum
# Paths
train_path, valid_path, test_path = args.train, args.val, args.test
model_path = args.save_dir
logs_path = args.expt_dir
pretrained_path = args.pretrain

# Logging
train_log_name = '{}-{}-{}-{}-{}-{}-{}-{}.train.log'.format(num_hidden, ','.join([str(word) for word in sizes]), activation, output_choice, batch_size, loss, opt, lr)
valid_log_name = '{}-{}-{}-{}-{}-{}-{}-{}.valid.log'.format(num_hidden, ','.join([str(word) for word in sizes]), activation, output_choice, batch_size, loss, opt, lr)
train_log = setup_logger('train-log', os.path.join(logs_path, train_log_name))
valid_log = setup_logger('valid-log', os.path.join(logs_path, valid_log_name))
# Load data
data = loadData(train_path, valid_path, test_path)
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                     data['valid']['X'], data['valid']['Y'],\
                                                     data['test']['X'], data['test']['Y'], 

# Initialize network
np.random.seed(1234)
network = Network(num_hidden, sizes, activation_choice = activation, output_choice = output_choice, loss_choice = loss)
model_name = '{}-{}-{}-{}-{}-{}-{}-{}.npy'.format(num_hidden, ','.join([str(word) for word in sizes]), activation, output_choice, batch_size, loss, opt, lr)
if pretrained_path != None:
    network.load(path = pretrained_path)
optimizer = Optimizers(network.theta.shape[0], opt, lr, momentum)
# Train
num_epochs = 1000
num_batches = int(float(train_X.shape[1]) / batch_size)
steps = 0
lr_min = 0.00001
loss_history = [np.inf]
prev_loss = np.inf
indices = np.arange(train_X.shape[1])
for epoch in range(num_epochs):
    steps = 0
    np.random.shuffle(indices)
    train_X, train_Y = train_X[:, indices], train_Y[indices]
    epoch_loss = []
    for batch in range(num_batches):
        start, end = batch * batch_size, (batch + 1) * batch_size
        x, y = train_X[:, range(start, end)], train_Y[range(start, end)]
        optimizer.opts[opt](network, x, y)
        grad_norm = np.linalg.norm(network.grad_theta)
        steps += batch_size
        if steps % train_X.shape[1] == 0 and steps != 0:
            y_pred, train_loss = network.forward(train_X, train_Y)
            error = network.performance(train_Y, y_pred)
            train_log.info('Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}'.format(epoch, steps, train_loss, error, optimizer.lr))
            y_pred, valid_loss = network.forward(valid_X, valid_Y)
            error = network.performance(valid_Y, y_pred)
            valid_log.info('Epoch {}, Step {}, Loss: {}, Error: {}, lr: {}'.format(epoch, steps, valid_loss, error, optimizer.lr))
            if valid_loss < min(loss_history):
                network.save(os.path.join(model_path, model_name))    
            loss_history.append(valid_loss)
    if anneal == True and valid_loss > prev_loss:
        network.load(path = os.path.join(model_path, model_name))
        if optimizer.lr > lr_min:
            optimizer.lr /= 2
        else:
            optimizer.lr = lr_min
    else:
        prev_loss = valid_loss

