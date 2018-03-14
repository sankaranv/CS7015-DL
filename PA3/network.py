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

class Network:

    def __init__(self, conv_sizes, dense_sizes, num_out, initializer='xavier', arch, lr):
        self.params = {}
        self.layers = {}
        self.arch = arch
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

    def model(self, x):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        c_conv = 0
        c_dense = 0
        c_pool = 0
        for i in range(len(arch)):
            if self.arch[i]=='conv':
                weight = 'Wc{}'.format(c_conv)
                bias = 'bc{}'.format(c_conv)
                layer = 'conv{}'.format(c_conv)
                x = conv2d(x, self.params[weight], self.params[bias])
                self.layers[layer] = x
                c_conv += 1
            elif self.arch[i]=='pool':
                layer = 'pool{}'.format(c_pool)
                x = pool2d(x)
                self.layers[layer] = x
                c_pool += 1
            elif self.arch[i]=='dense':
                weight = 'Wd{}'.format(c_dense)
                bias = 'bd{}'.format(c_dense)
                layer = 'fc{}'.format(c_pool)
                x = dense(x, self.params[weight], self.params[bias])
                self.layers[layer] = x
                c_dense += 1
            elif self.arch[i]=='out':
                logits = tf.add(tf.matmul(x, self.params['Wout']), self.params['bout'])
                self.layers['out'] = logits
                return self.layers['out']
            else:
                print "Invalid architecture!"
                sys.exit(1)

    def predict(self,x,y,lr):
        logits = model(x)
        y_pred =  tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(correct_pred)

    def train():
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                steps = 0
                for batch in range(num_batches):
                 #batch_x, batch_y
                 sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
                 loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y)
