import numpy as np
import h5py
from data import loadData

def sigmoid(x):
    #print 1.0 / (1.0 + np.exp(-x))
    return 1.0 / (1.0 + np.exp(-x))

class RBM:

    def __init__(self, num_V, num_H):

        self.num_V = num_V
        self.num_H = num_H

		#Weights and Biases
		#Refer Hinton's practical guide to training RBMs
        #self.W = np.random.normal(0.0, 0.001, (num_V,num_H))
        self.W = np.zeros((num_V, num_H))
        self.b = np.zeros((1, num_V))
        self.c = np.zeros((1, num_H))

    def block_gibbs_sample_VgivenH(self, init_H):
        # Probability of V being 1 given H
        expected_V = sigmoid(np.dot(init_H, self.W.T) + self.b)
        #Bernoulli Sampling
        batch_size, size = expected_V.shape
        rand_vec = np.random.rand(batch_size, size)
        sample_V = np.int32(rand_vec < expected_V)
        return sample_V

    def block_gibbs_sample_HgivenV(self, init_V):
        # Probability of H being 1
        expected_H = sigmoid(np.dot(init_V, self.W) + self.c)
		#Bernoulli Sampling
        batch_size, size = expected_H.shape
        rand_vec = np.random.rand(batch_size, size)
        sample_H = np.int32(rand_vec < expected_H)
        return sample_H

    def train_contrastive_divergence(self, data, k = 20, batch_size = 32, lr = 0.01, epochs = 10, save_path = None):
        num_batches = data.shape[0] / batch_size
        loss_history = [np.inf]
        for epoch in range(epochs):
            print 'Epoch-{}'.format(epoch + 1)
            for batch in range(num_batches):
                V_data = data[batch * batch_size : (batch + 1) * batch_size, :]
                V_model = np.copy(V_data)
                for t in range(k):
                    H_model = self.block_gibbs_sample_HgivenV(V_model)
                    V_model = self.block_gibbs_sample_VgivenH(H_model)
                grad_W = np.dot(V_data.T, sigmoid(np.dot(V_data, self.W) + self.c))\
                                - np.dot(V_model.T, sigmoid(np.dot(V_model, self.W) + self.c))
                grad_b = (V_data - V_model).sum(axis = 0)
                grad_c = (sigmoid(np.dot(V_data, self.W) + self.c) - sigmoid(np.dot(V_model, self.W) + self.c)).sum(axis = 0)
                self.W += 1.0 / batch_size * lr * grad_W
                self.b += 1.0 / batch_size * lr * grad_b
                self.c += 1.0 / batch_size * lr * grad_c
            H_model = self.block_gibbs_sample_HgivenV(data)
            V_model = self.block_gibbs_sample_VgivenH(H_model)
            loss = 1.0 / data.shape[0] * np.power((V_model - data), 2).sum()
            if loss < np.min(loss_history):
                self.saveModel(save_path)
            else:
                lr *= 0.5
            loss_history.append(loss)
            print 'Epoch {}: {}'.format(epoch + 1, loss)

    def saveModel(self, save_path):
        with h5py.File(save_path, 'w') as f:
            f.create_dataset(data = self.W, name = 'W', shape = self.W.shape, dtype = self.W.dtype)
            f.create_dataset(data = self.b, name = 'b', shape = self.b.shape, dtype = self.b.dtype)
            f.create_dataset(data = self.c, name = 'c', shape = self.c.shape, dtype = self.c.dtype)

    def loadModel(self, load_path):
        with h5py.File(load_path, 'r') as f:
            self.W = f['W'][:]
            self.b = f['b'][:]
            self.c = f['c'][:]

if __name__ == '__main__':
    train_data = loadData('data/train.csv', 'data/val.csv', 'data/test.csv')['train']['X']
    rbm = RBM(784, 100)
    # Hyperparams
    k = 1
    batch_size = 50
    lr = 0.1
    epochs = 20
    save_path = '{}-{}-{}'.format(k, batch_size, epochs)
    rbm.train_contrastive_divergence(train_data, k = k, batch_size = batch_size, lr = lr, epochs = epochs,\
                                     save_path = save_path)
