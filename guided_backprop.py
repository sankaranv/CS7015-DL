import sys, os
import shutil
import resource
import argparse
import logging
from helpers import setup_logger

import numpy as np
import tensorflow as tf

from data import loadData
from network import CNN
from augment import Augment
from modelparser import loadArch

@tf.RegisterGradient("GuidedRelu")
	def _GuidedReluGrad(op, grad):
		gate_g = tf.cast(grad > 0, "float32")
	    gate_y = tf.cast(op.outputs[0] > 0, "float32")
		return gate_y * gate_g * grad

parser = argparse.ArgumentParser(description='Train the CNN')
parser.add_argument('--expt_dir', default='./logs',
                    help='save dir for experiment logs')
parser.add_argument('--train', default='./data/train.csv',
                    help='path to training set')
parser.add_argument('--val', default='./data/valid.csv',
                    help='path to validation set')
parser.add_argument('--test', default='./data/test.csv',
                    help='path to test set')
parser.add_argument('--save_dir', default='./models',
                    help='path to save model')
parser.add_argument('--arch', default='models/cnn.json',
                    help = 'path to model architecture')
parser.add_argument('--model_name', default = 'model',
                    help = 'name of the model to save logs, weights')
parser.add_argument('--lr', default = 0.001,
                    help = 'learning rate')
parser.add_argument('--init', default = '1',
                    help = 'initialization')
parser.add_argument('--batch_size', default = 20,
                    help = 'batch_size')
args = parser.parse_args()

# Load data
train_path, valid_path, test_path = args.train, args.val, args.test
logs_path = args.expt_dir
model_path, model_arch, model_name = args.save_dir, args.arch, args.model_name
model_path = os.path.join(model_path, model_name)
if not os.path.isdir(model_path):
    os.mkdir(model_path)
lr, batch_size, init = float(args.lr), int(args.batch_size), int(args.init)

data = loadData(train_path, valid_path, test_path)
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                     data['valid']['X'], data['valid']['Y'],\
                                                     data['test']['X'], data['test']['Y'],

# Load architecture
arch = loadArch(model_arch)

# Logging
train_log_name = '{}.train.log'.format(model_name)
valid_log_name = '{}.valid.log'.format(model_name)
train_log = setup_logger('train-log', os.path.join(logs_path, train_log_name))
valid_log = setup_logger('valid-log', os.path.join(logs_path, valid_log_name))

# GPU config
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)

# Train
num_epochs = 50
num_batches = int(float(train_X.shape[0]) / batch_size)
steps = 0
patience = 50
early_stop = 0

model = CNN(arch, session, logs_path, init, lr)
model.load(os.path.join(model_path, 'model'))

graph = model.graph
graph_def = graph.as_graph_def()
guided_graph = tf.Graph()
with self.guided_graph.as_default():
	self.guided_sess = tf.Session(graph = self.guided_graph)
	with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
		tf.import_graph_def(graph_def, name='')




