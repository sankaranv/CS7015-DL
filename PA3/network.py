import tensorflow as tf

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def pool2d(x,k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')



class Network:

    def __init__(self, conv_sizes, dense_sizes, num_out, initializer='xavier'):
        self.params = {}
        # Initializers
        if initializer='he':
            init = tf.contrib.layers.variance_scaling_initializer()
        else:
            init = tf.contrib.layers.xavier_initializer()
        # Conv weights
        for i in range(len(conv_sizes)):
            weight = 'Wc{}'.format(i)
            self.params[weight] = tf.get_variable(name = weight, shape = conv_sizes[i], initializer = init)
            bias = 'bc{}'.format(i)
            self.params[bias] = tf.get_variable(name = bias, shape = [conv_sizes[i][3]], initializer = init)
        # Dense weights
        for i in range(1,len(dense_sizes)):
            weight = 'Wd{}'.format(i)
            self.params[weight] = tf.get_variable(name = weight, shape = [dense_sizes[i],dense_sizes[i-1]], initializer = init)
            bias = 'bd{}'.format(i)
            self.params[bias] = tf.get_variable(name = weight, shape = [dense_sizes[i],1], initializer = init)
        # Output layer weight
        self.params['Wout'] = tf.get_variable(name = 'Wout', shape = [num_out,dense_sizes[-1]], initializer = init)
        self.params['bout'] = tf.get_variable(name = 'bout', shape = [num_out,1], initializer = init)
