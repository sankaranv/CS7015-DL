import numpy as np
import h5py
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from data import loadData

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class RBM:

    def __init__(self, num_V, num_H):

        self.num_V = num_V
        self.num_H = num_H

		#Weights and Biases
		#Refer Hinton's practical guide to training RBMs
        self.W = np.random.normal(0.0, 0.001, (num_V,num_H))
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
        # Probability of H being 1 given V
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
                lr *= 1.0
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

    def saveEmbeddings(self, data, labels, save_path):
        embedding = self.block_gibbs_sample_HgivenV(data)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset(data = embedding, name = 'h', shape = embedding.shape, dtype = embedding.dtype)
            f.create_dataset(data = labels, name = 'l', shape = labels.shape, dtype = labels.dtype)
        return embedding

    def loadEmbeddings(self, load_path):
        with h5py.File(load_path, 'r') as f:
            embeddings = f['h'][:]
            labels = f['l'][:]
        return embeddings, labels

def plotTSNE(data, save_path):
    embeddings, labels = data
    colors = np.array(['green', 'violet', 'yellow', 'blue', 'black', 'purple', 'violet', 'pink', 'orange', 'cyan'])
    markers = np.array(['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot'])
    tsne_embed = TSNE(n_components = 2, learning_rate = 10.0, n_iter = 1000, init = 'pca').fit_transform(embeddings)
    data = dict()
    fig = plt.figure(0)
    ax = fig.gca()
    ax.set_xlim([-40, 50])
    ax.set_ylim([-40, 40])
    ax.set_title('TSNE embeddings')
    for i in range(tsne_embed.shape[0]):
        if labels[i] not in data.keys():
            data[labels[i]] = []
        data[labels[i]].append(tsne_embed[i, :])
    for label in range(10):
        data[label] = np.stack(data[label], axis = 0)
        x, y = data[label][:, 0], data[label][:, 1]
        ax.scatter(x, y, color = colors[label], label = markers[label], s = 1.0)
    ax.legend()
    plt.savefig(save_path)

if __name__ == '__main__':
    MODE = 'TRAIN,VIZ,TEST'
    data = loadData('data/train.csv', 'data/val.csv', 'data/test.csv')

    train_data = data['train']['X']
    test_data, labels = data['test']['X'], data['test']['Y']

    # Hyperparams
    k = 1
    batch_size = 1
    lr = 0.01
    epochs = 10
    hidden_size = 300
    visible_size = train_data.shape[1]

    rbm = RBM(visible_size, hidden_size)

    if 'TRAIN' in MODE:
        save_path = 'models\\{}-{}-{}-{}.h5'.format(hidden_size, k, batch_size, epochs)
        rbm.train_contrastive_divergence(train_data, k = k, batch_size = batch_size, lr = lr, epochs = epochs,\
                                     save_path = save_path)
    if 'TEST' in MODE:
        load_path = 'models\\{}-{}-{}-{}.h5'.format(hidden_size, k, batch_size, epochs)
        rbm.loadModel(load_path)
        rbm.saveEmbeddings(test_data, labels, 'embeddings\\{}-{}-{}-{}.h5'.format(hidden_size, k, batch_size, epochs))

    if 'VIZ' in MODE:
        data = rbm.loadEmbeddings('embeddings\\{}-{}-{}-{}.h5'.format(hidden_size, k, batch_size, epochs))
        plotTSNE(data, 'plots\\{}-{}-{}-{}.jpg'.format(hidden_size, k, batch_size, epochs))
