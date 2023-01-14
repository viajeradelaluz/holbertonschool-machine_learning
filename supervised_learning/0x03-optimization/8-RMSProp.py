#!/usr/bin/env python3
"""
Module with task 8. RMSProp Upgraded
"""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm.

    :param loss: loss of the network.
    :param alpha: learning rate.
    :param beta2: RMSProp weight.
    :param epsilon: small number to avoid division by zero.

    :return: RMSProp optimization operation.
    """
    rms = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)

    return rms.minimize(loss)
