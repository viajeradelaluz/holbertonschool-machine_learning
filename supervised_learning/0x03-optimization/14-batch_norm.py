#!/usr/bin/env python3
"""
Module with task 14. Batch Normalization Upgraded
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow.

    :param prev: tensor
    :param n: number of nodes in the layer to be created
    :param activation: activation function to use on the output of the layer

    :return: tensor of the activated output for the layer
    """
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)
    epsilon = 1e-8

    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    Z = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])

    batch_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        epsilon,
    )

    return activation(batch_norm)
