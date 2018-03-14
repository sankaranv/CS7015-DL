import os
import numpy as np

def loadFromCSV(fname, is_test = False):
    lines = [line.strip().split(',') for line in open(fname, 'r').readlines()][1 : ]
    size = len(lines)
    dim_X, dim_Y = 28, 28
    dim = dim_X * dim_Y
    scale = 255.0
    X, Y = np.zeros((size, dim_X, dim_Y), dtype = np.float32), np.zeros((size), dtype = np.int32)
    for index, line in enumerate(lines):
        if is_test == False:
            Y[index] = np.int32(line[-1])
        X[index][:, :] = np.float32(line[1 : 1 + dim]).reshape((dim_X, dim_Y)) / scale
    print 'Loaded data of shape', X.shape, Y.shape
    return X, Y

def normalizationParams(X):
    mean = X.mean(axis = 0)
    stddev = np.sqrt(np.mean((X - mean) * (X - mean), axis = 0))
    return mean, stddev

def loadData(train_path, valid_path, test_path):
    stages = ['train', 'valid', 'test']
    normalize = True
    data_path = 'data'
    if os.path.isfile(os.path.join(data_path, 'train_X.npy')) == False:
        train_X, train_Y = loadFromCSV(train_path)
        valid_X, valid_Y = loadFromCSV(valid_path)
        test_X, test_Y   = loadFromCSV(test_path)

        data = {'train' : {'X' : train_X, 'Y' : train_Y},\
                'valid' : {'X' : valid_X, 'Y' : valid_Y},\
                'test'  : {'X' : test_X, 'Y'  : test_Y}}
    
        # Mean-variance normalization of features
        if normalize == True:
            mean, stddev = normalizationParams(train_X)
            for stage in stages:
                data[stage][0] = (data[stage]['X'] - mean) / stddev

        # Shuffle training data
        indices = np.arange(data['train']['X'].shape[0])
        np.random.seed(1234)
        np.random.shuffle(indices)
        data['train']['X'], data['train']['Y'] = data['train']['X'][indices], data['train']['Y'][indices]

        for stage in stages:
            np.save(os.path.join(data_path, '{}_X.npy'.format(stage)), data[stage]['X'])
            np.save(os.path.join(data_path, '{}_Y.npy'.format(stage)), data[stage]['Y'])

    data = {'train' : {'X' : None, 'Y' : None,},\
            'valid' : {'X' : None, 'Y' : None,},\
            'test'  : {'X' : None, 'Y' : None}}

    # Load data, return transpose, shape of returned array is (dimension, num_points) : each column is a data-point
    for stage in stages:
        data[stage]['X'] = np.load(os.path.join(data_path, '{}_X.npy'.format(stage))).T
        data[stage]['Y'] = np.load(os.path.join(data_path, '{}_Y.npy'.format(stage))).T

    print 'Loaded train-val-test data'
    
    return data

if __name__ == '__main__':

    loadData('data/train.csv', 'data/val.csv', 'data/test.csv')
    pass
