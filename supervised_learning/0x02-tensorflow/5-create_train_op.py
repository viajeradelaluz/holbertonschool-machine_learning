#!/usr/bin/env python3
"""
Module that creates the training operation for the network.
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network.
    :param loss: loss of the network's prediction.
    :param alpha: learning rate.
    :return: an operation that trains the network using gradient descent.
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
