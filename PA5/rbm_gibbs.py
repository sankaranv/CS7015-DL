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

    def train_gibbs_sampling(self, data, k = 20, r = 20, lr = 0.01, epochs = 10, save_path = None):
        loss_history = [np.inf]
        for epoch in range(epochs):
            print 'Epoch-{}'.format(epoch + 1)
            for index in range(data.shape[0]):
                V_history = []
                V_data = data[index, :].reshape((1, self.num_V))
                V_model = np.random.randint(0, 1, size = V_data.shape)
                # Gibbs sampling
                for t in range(k+r):
                    H_model = self.block_gibbs_sample_HgivenV(V_model)
                    V_model = self.block_gibbs_sample_VgivenH(H_model)
                    if(t >= k):
                        V_history.append(V_model)
                V_history = np.concatenate(V_history, axis = 0)
                W_sum = 1.0 / r * np.dot(V_history.T, sigmoid(np.dot(V_history, self.W) + self.c))
                b_sum = 1.0 / r * V_history.sum(axis = 0)
                c_sum = 1.0 / r * (sigmoid(np.dot(V_history, self.W) + self.c)).sum(axis = 0)

                grad_W = np.dot(V_data.T, sigmoid(np.dot(V_data, self.W) + self.c)) - W_sum
                grad_b = (V_data).sum(axis = 0) - b_sum
                grad_c = (sigmoid(np.dot(V_data, self.W) + self.c)).sum(axis = 0) - c_sum

                self.W += lr * grad_W
                self.b += lr * grad_b
                self.c += lr * grad_c

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
    colors = np.array(['green', 'red', 'yellow', 'blue', 'black', 'purple', 'violet', 'pink', 'orange', 'cyan'])
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
    MODE = 'TEST-VIZ'
    data = loadData('data/train.csv', 'data/val.csv', 'data/test.csv')

    train_data = data['train']['X']
    test_data, labels = data['test']['X'], data['test']['Y']

    # Hyperparams
    k = 20
    r = 20
    lr = 0.01
    epochs = 10
    hidden_size = 200
    visible_size = train_data.shape[1]

    rbm = RBM(visible_size, hidden_size)

    if 'TRAIN' in MODE:
        save_path = 'models\\gibbs-{}-{}-{}-{}.h5'.format(hidden_size, k, r, epochs)
        rbm.train_gibbs_sampling(train_data, k = k, r = r, lr = lr, epochs = epochs,\
                                     save_path = save_path)
    if 'TEST' in MODE:
        load_path = 'models\\gibbs-{}-{}-{}-{}.h5'.format(hidden_size, k, r, epochs)
        rbm.loadModel(load_path)
        rbm.saveEmbeddings(test_data, labels, 'embeddings\\gibbs-{}-{}-{}-{}.h5'.format(hidden_size, k, r, epochs))

    if 'VIZ' in MODE:
        data = rbm.loadEmbeddings('embeddings\\gibbs-{}-{}-{}-{}.h5'.format(hidden_size, k, r, epochs))
        plotTSNE(data, 'plots\\gibbs-{}-{}-{}-{}.jpg'.format(hidden_size, k, r, epochs))
