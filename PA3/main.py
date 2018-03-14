import tensorflow as tf
from network import CNN

# Parse args

def parse_arch():
    pass

with tf.Graph().as_default(), tf.Session() as session:
    model = CNN(conv_sizes, dense_sizes, num_out, arch, session)
    
