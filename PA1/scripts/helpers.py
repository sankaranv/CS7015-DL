import numpy as np
import logging
formatter = logging.Formatter('%(message)s')


def activation_function(x, activation = 'sigmoid'):
    if (activation == 'sigmoid'):
        return 1 / (1 + np.exp(-x))
    elif (activation == 'tanh'):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    elif activation == 'relu':
        x[np.where(x < 0)] = 0.0
        return x
    else:
        return x

def output_function(x, activation = 'softmax'):
    if (activation == 'softmax'):
        x = np.exp(x - np.max(x, axis = 0))  # Normalization for numerical stability, from CS231n notes
        return x / np.sum(x, axis=0)
    if (activation == 'sigmoid'):
        return 1 / (1 + np.exp(-x))
    else:
        return x

def loss_function(y_true, y_pred, loss = 'ce'):
    batch_size = y_true.shape[0]
    if loss == 'sq':
        e_y = np.zeros_like(y_pred)
        e_y[y_true, range(batch_size)] = 1
        return (1.0 / (2.0 * batch_size)) * np.sum((e_y - y_pred)**2)
    if loss == 'ce':
        return (-1.0 / batch_size) * np.log(y_pred[y_true, range(batch_size)]).sum()


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode = 'w')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
