#!/usr/bin/env python3
"""
Module that creates a layer of a neural network using dropout.
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout

    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation: activation function that should be used on theh layer
    :param keep_prob: probability that a node will be kept

    :return: the output of the new layer
    """

    dropout = tf.layers.Dropout(rate=keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=dropout,
    )

    return layer(prev)
