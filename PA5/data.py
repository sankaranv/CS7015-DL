import os
import numpy as np


def loadFromCSV(fname, is_test = False):
    lines = [line.strip().split(',') for line in open(fname, 'r').readlines()][1 : ]
    size = len(lines)
    dim = 784
    scale = 255.0
    X, Y = np.zeros((size, dim), dtype = np.float32), np.zeros((size), dtype = np.int32)
    for index, line in enumerate(lines):
        if is_test == False:
            Y[index] = np.int32(line[-1])
        X[index][:] = np.float32([0 if float(pixel) < 127 else 1 for pixel in line[1 : 1 + dim]])
    if is_test == True:
        with open('data\\test_labels.csv', 'r') as f:
            Y = np.int32([int(line.strip().split(',')[1]) for line in f.readlines()[1:]])
            print Y
    print 'Loaded data of shape', X.shape, Y.shape
    return X, Y

def loadData(train_path, valid_path, test_path):
    stages = ['train', 'test']
    normalize = True
    data_path = 'data'
    if os.path.isfile(os.path.join(data_path, 'train_X.npy')) == False:
        train_X, train_Y = loadFromCSV(train_path)
        valid_X, valid_Y = loadFromCSV(valid_path)
        test_X, test_Y   = loadFromCSV(test_path, is_test = True)

        data = {'train' : {'X' : np.concatenate([train_X, valid_X], axis = 0), 'Y' : np.concatenate([train_Y, valid_Y], axis = 0)},\
                'test'  : {'X' : test_X, 'Y'  : test_Y}}

        # Shuffle training data
        indices = np.arange(data['train']['X'].shape[0])
        np.random.seed(1234)
        np.random.shuffle(indices)
        data['train']['X'], data['train']['Y'] = data['train']['X'][indices], data['train']['Y'][indices]

        for stage in stages:
            np.save(os.path.join(data_path, '{}_X.npy'.format(stage)), data[stage]['X'])
            np.save(os.path.join(data_path, '{}_Y.npy'.format(stage)), data[stage]['Y'])

    data = {'train' : {'X' : None, 'Y' : None,},\
            'test'  : {'X' : None, 'Y' : None}}

    # Load data, shape of returned array is (num_points, dimension) : each row is a data-point
    for stage in stages:
        data[stage]['X'] = np.load(os.path.join(data_path, '{}_X.npy'.format(stage)))
        data[stage]['Y'] = np.load(os.path.join(data_path, '{}_Y.npy'.format(stage)))

    print 'Loaded train-val-test data'

    return data

if __name__ == '__main__':

    loadData('data/train.csv', 'data/val.csv', 'data/test.csv')
    pass
