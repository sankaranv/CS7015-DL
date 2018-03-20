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
from modelparser import loadArch

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

# Test model on data after every epoch
def test(model, x, y, epoch, batch_size, is_train):
    print 'Testing for epoch {}'.format(epoch)
    num_batches = x.shape[0] / batch_size
    loss, acc = np.zeros((num_batches)), np.zeros((num_batches))
    for batch in range(num_batches):
        start, end = batch * batch_size, (batch + 1) * batch_size
        batch_x, batch_y = x[start : end], y[start : end]
        idx = epoch*num_batches + batch
        loss[batch], acc[batch] = model.performance(batch_x, batch_y, is_train, idx)
    return np.mean(loss), np.mean(acc)

def train():
    # Parse args
    parser = argparse.ArgumentParser(description='Train the CNN')
    parser.add_argument('--expt_dir', default='./logs',
                        help='save dir for experiment logs')
    parser.add_argument('--train', default='./data',
                        help='path to training set')
    parser.add_argument('--val', default='./data',
                        help='path to validation set')
    parser.add_argument('--test', default='./data',
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    # Train
    num_epochs = 20
    num_batches = int(float(train_X.shape[0]) / batch_size)
    steps = 0
    patience = 5
    early_stop=0

    with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as session:
        model = CNN(arch, session, logs_path, init, lr)
        loss_history = [np.inf]
        for epoch in range(num_epochs):
            print 'Epoch {}'.format(epoch)
            steps = 0
            indices = np.arange(train_X.shape[0])
            np.random.shuffle(indices)
            train_X, train_Y = train_X[indices], train_Y[indices]
            for batch in range(num_batches):
                start, end = batch * batch_size, (batch + 1) * batch_size
                x, y = train_X[range(start, end)], train_Y[range(start, end)]
                try:
                    model.step(x,y)
                except MemoryError:
                    print 'Memory error in step'
                    exit()
                steps += batch_size
                if steps % train_X.shape[0] == 0 and steps != 0:
                    try:
                        train_loss, train_acc = test(model, train_X, train_Y, epoch, batch_size, True)
                    except MemoryError:
                        print 'Memory error in test for train'
                        exit()
                    train_log.info('Epoch {}, Step {}, Loss: {}, Accuracy: {}, lr: {}'.format(epoch, steps, train_loss, train_acc, model.lr))
                    try:
                        valid_loss, valid_acc = test(model, valid_X, valid_Y, epoch, batch_size, False)               
                    except MemoryError:
                        print 'Memory error in test for valid'
                        exit()
                    valid_log.info('Epoch {}, Step {}, Loss: {}, Accuracy: {}, lr: {}'.format(epoch, steps, valid_loss, valid_acc, model.lr))
                    if valid_loss < min(loss_history):
                        model.save(os.path.join(model_path, model_name))
                        early_stop = 0
                    early_stop += 1
                    if (early_stop >= patience):
                        print "No improvement in validation loss for " + str(patience) + " steps - stopping training!"
                        print("Optimization Finished!")
                        return 1
                    loss_history.append(valid_loss)
        print("Optimization Finished!")

if __name__ == '__main__':
    memory_limit()
    train()
			
