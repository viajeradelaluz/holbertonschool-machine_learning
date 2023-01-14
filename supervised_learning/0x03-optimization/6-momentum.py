#!/usr/bin/env python3
"""
Module with task 6. Momentum upgraded
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm.

    :param loss: the loss of the network
    :param alpha: the learning rate
    :param beta1: the momentum weight

    :return: the momentum optimization operation
    """
    momentum = tf.train.MomentumOptimizer(alpha, beta1)
    return momentum.minimize(loss)
