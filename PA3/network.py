import tensorflow as tf
import sys

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def pool2d(x,k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def dense(x, W, b):
    fc = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, W), b)
    return tf.nn.relu(fc)

class CNN:

    def __init__(self, conv_sizes, dense_sizes, num_out, arch, session, initializer='xavier', lr = 0.001):
        self.params = {}
        self.layers = {}
        self.arch = arch
        self.lr = lr
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

        # Build the TensorFlow graph
        self.sess = session
        self.build_graph()

    def build_graph(self):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='input_data')
        y = tf.placeholder(tf.float32, shape=[None, 10], name='input_labels')
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        c_conv = 1
        c_dense = 1
        c_pool = 1

        for i in range(len(arch)):
            prev_layer_idx = ''

            if self.arch[i]=='conv':
                weight = 'Wc{}'.format(c_conv)
                bias = 'bc{}'.format(c_conv)
                layer_idx = 'conv{}'.format(c_conv)
                if i == 0:
                    self.layers[layer_idx] = conv2d(x, self.params[weight], self.params[bias])
                else
                    self.layers[layer_idx] = conv2d(self.layers[prev_layer_idx], self.params[weight], self.params[bias])
                c_conv += 1
                prev_layer_idx = layer_idx

            elif self.arch[i]=='pool':
                layer_idx = 'pool{}'.format(c_pool)
                self.layers[layer_idx] = pool2d(self.layers[prev_layer_idx])
                c_pool += 1
                prev_layer_idx = layer_idx

            elif self.arch[i]=='dense':
                weight = 'Wd{}'.format(c_dense)
                bias = 'bd{}'.format(c_dense)
                layer_idx = 'fc{}'.format(c_pool)
                self.layers[layer_idx] = dense(self.layers[prev_layer_idx], self.params[weight], self.params[bias])
                c_dense += 1

            elif self.arch[i]=='out':
                logits = tf.add(tf.matmul(x, self.params['Wout']), self.params['bout'])
                self.layers['out'] = logits
                y_pred =  tf.nn.softmax(logits)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

            else:
                print "Invalid architecture!"
                sys.exit(1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(correct_pred)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def step(self, batch_x, batch_y):
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y)

    def load_data(self):

    def train(self):
        
