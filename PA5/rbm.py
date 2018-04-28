import numpy as np

class RBM:

	def __init__(self, num_V, num_H, lr):
		
		self.num_V = num_V
		self.num_H = num_H
		self.lr = lr
		#Weights and Biases
		#Refer Hinton's practical guide to training RBMs
		self.W = np.random.normal(0.0,0.01,(num_V,num_H))
		self.b = np.zeros(num_V)
		self.c = np.zeros(num_H)

	def train(self, data, num_epochs, lr):

		num_examples = data.shape[0]

	def contrastive_divergence(self, k, V):
		
		#Positive Sample
		pos_H = block_gibbs_sample(mode = 'V', init_V = V)
		pos_V = V

		#Negative Sample
		neg_V = np.copy(V)
		for i in range(k):
			neg_H=block_gibbs_sample(mode = 'V', init_V = neg_V)
			neg_V=block_gibbs_sample(mode = 'H', init_H = neg_H)	
		neg_H=block_gibbs_sample(mode = 'V', init_V = neg_V)

		self.W += self.lr * np.dot(pos_H, pos_V.T)
		self.W -= self.lr * np.dot(neg_H, neg_V.T) 
		self.b += self.lr * pos_V
		self.b -= self.lr * neg_V
		self.c += self.lr * pos_H
		self.c -= self.lr * neg_H



		
