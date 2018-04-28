import numpy as np

class RBM:

	def __init__(self, num_V, num_H):
		
		self.num_V = num_V
		self.num_H = num_H

		#Weights and Biases
		#Refer Hinton's practical guide to training RBMs
		self.W = np.random.normal(0.0,0.01,(num_V,num_H))
		self.b = np.zeros(num_V)
		self.c = np.zeros(num_H)

	def train_contrastive_divergence(self, k, data, batch_size, lr):
		
		num_iter = data.shape[0]/batch_size
		for d in range(num_iter):
			V0 = data[d*batch_size:(d+1)*batch_size]		
			#Positive Sample
			pos_H = block_gibbs_sample_HgivenV(init_V = V0)
			pos_V = V0
			#Negative Sample
			neg_V = np.copy(V0)
			for i in range(k):
				neg_H=block_gibbs_sample_HgivenV(init_V = neg_V)
				neg_V=block_gibbs_sample_VgivenH(init_H = neg_H)	
			neg_H=block_gibbs_sample_HgivenV(init_V = neg_V)

			self.W += self.lr * (np.dot(pos_H, pos_V.T) - np.dot(neg_H, neg_V.T))
			self.b += self.lr * (pos_V - neg_V)
			self.c += self.lr * (pos_H - neg_H)


	def block_gibbs_sample_VgivenH(init_H):
		expected_V = np.dot(init_H.T,self.W).T+self.b
        #Sigmoid
        expected_V = np.exp(expected_V)
        expected_V /= (1.+expected_V)
        #Bernoulli Sampling
        rand_vec = np.random.rand(self.num_V,1)
        sample_V = int(rand_vec<expected_V)
        return sample_V


	def block_gibbs_sample_HgivenV(init_V):
		expected_H = np.dot(self.W, init_V)+self.c
		#Sigmoid
		expected_H = np.exp(expected_H)
		expected_H /= (1.+expected_H)
		#Bernoulli Sampling
		rand_vec = np.random.rand(self.num_H,1)
		sample_H = int(rand_vec<expected_H)
		return sample_H


		
