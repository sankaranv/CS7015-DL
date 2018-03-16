import tensorflow as tf
import sys

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def pool2d(x, k = 2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def dense(x, W, b):
    fc = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, W), b)
    return tf.nn.relu(fc)

class CNN:

    def __init__(self, arch, session, logs_path, initializer='xavier', lr = 0.001):
        self.params = {}
        self.layers = {}
        self.arch = arch
        self.lr = lr
        self.logs_path = logs_path
        # Initializers
        if initializer=='he':
            init = tf.contrib.layers.variance_scaling_initializer()
        else:
            init = tf.contrib.layers.xavier_initializer()
        for item in self.arch:
            layer = item['name']
            params = item['params']
            if params == 'NONE':
                continue
            else:
                weight = params['weight']['name']
                shape = params['weight']['shape']
                self.params[weight] = tf.get_variable(name = weight, shape = shape, initializer = init)
                bias = params['bias']['name']
                shape = params['bias']['shape']
                self.params[bias] = tf.get_variable(name = bias, shape = shape, initializer = init)

        # Build the TensorFlow graph
        self.sess = session
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='input_data')
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name='input_labels')

        prev_layer = ''
        for index, item in enumerate(self.arch):
            layer = item['name']
            if item['params'] != 'NONE':
                params = item['params']
                weight = params['weight']['name']
                bias = params['bias']['name']

            if 'input' in layer:
                self.layers[layer] = tf.reshape(self.x, shape=[-1, 28, 28, 1])

            elif 'conv' in layer:
                padding, stride = item['padding'], item['stride']
                self.layers[layer] = conv2d(self.layers[prev_layer], self.params[weight], self.params[bias], stride)

            elif 'pool' in layer:
                padding, stride = item['padding'], item['stride']
                self.layers[layer] = pool2d(self.layers[prev_layer])

            elif 'reshape' in layer:
                continue

            elif 'dropout' in layer:
                prob = item['prob']
                self.layers[layer] = tf.nn.dropout(self.layers[prev_layer], keep_prob=prob)

            elif 'batchnorm' in layer:
                self.layers[layer] = tf.contrib.layers.batch_norm(self.layers[prev_layer])

            elif 'fc' in layer:
                self.layers[layer] = dense(self.layers[prev_layer], self.params[weight], self.params[bias])

            elif 'output' in layer:
                logits = tf.add(tf.matmul(self.layers[prev_layer], self.params['Wout']), self.params['bout'])
                self.layers[layer] = logits
                y_pred =  tf.nn.softmax(logits)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

            else:
                print "Invalid architecture!"
                sys.exit(1)

            print 'Adding Layer-{} : {}, Shape = {}'.format((index + 1), layer, self.layers[layer].get_shape())

            prev_layer = layer

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train_op = optimizer.minimize(self.loss)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        # TensorBoard summaries
        self.train_loss = tf.summary.scalar("train_loss", self.loss)
        self.valid_loss = tf.summary.scalar("valid_loss", self.loss)
        self.train_acc = tf.summary.scalar("train_accuracy", self.accuracy)
        self.valid_acc = tf.summary.scalar("valid_accuracy", self.accuracy)
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())

    def step(self, batch_x, batch_y):
        self.sess.run(self.train_op, feed_dict = {self.x: batch_x, self.y: batch_y})

    def performance(self, batch_x, batch_y, is_train, idx):
        if is_train == True:
            loss, acc, loss_summary, accuracy_summary = self.sess.run([self.loss, self.accuracy, self.train_loss, self.train_acc],
                                                                       feed_dict={self.x: batch_x, self.y: batch_y})
        else:
            loss, acc, loss_summary, accuracy_summary = self.sess.run([self.loss, self.accuracy, self.valid_loss, self.valid_acc],
                                                                       feed_dict={self.x: batch_x, self.y: batch_y})
        self.summary_writer.add_summary(loss_summary, idx)
        self.summary_writer.add_summary(accuracy_summary, idx)
        return loss, acc

    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
