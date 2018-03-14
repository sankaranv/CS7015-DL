import tensorflow as tf

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def pool2d(x,k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')



class Network:

    def __init__(self, conv_sizes, fc_sizes):
        self.params = {}
        for i in range(len(conv_sizes)):
            weight = 'Wc{}'.format(i)
            self.params[weight] = tf.Variable(tf.random_normal(conv_sizes[i]))
            bias = 'bc{}'.format(i)
            self.params[bias] = tf.Variable(tf.random_normal([conv_sizes[i][3]]))
