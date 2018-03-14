import tensorflow as tf
import sys
import os
import argparse
import logging

from network import CNN
from data import loadData

# Parse args

parser.add_argument('--train', default='./data',
                    help='path to training set')
parser.add_argument('--val', default='./data',
                    help='path to validation set')
parser.add_argument('--test', default='./data',
                    help='path to test set')

def parse_arch():
    pass

# Load data
train_path, valid_path, test_path = args.train, args.val, args.test
data = loadData(train_path, valid_path, test_path)
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                     data['valid']['X'], data['valid']['Y'],\
                                                     data['test']['X'], data['test']['Y'],
# Train
num_epochs = 1000
batch_size = 20
num_batches = int(float(train_X.shape[1]) / batch_size)
steps = 0

with tf.Graph().as_default(), tf.Session() as session:
    model = CNN(conv_sizes, dense_sizes, num_out, arch, session)
