#!/usr/bin/env python3
"""
Module that creates a tensorflow layer that includes L2 regularization.
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization:

    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes the new layer should contain
    :param activation: activation function that should be used
    :param lambtha: L2 regularization parameter

    :return: the output of the new layer
    """

    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    L2 = tf.keras.regularizers.L2(lambtha)

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=L2,
    )

    return layer(prev)
