#!/usr/bin/env python3
"""
Module to create a layer
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Creates a layer
    :param prev: the tensor output of the previous layer
    :param n: the number of nodes in the layer to create
    :param activation: is the activation function that the layer should use
    :return: the tensor output of the layer
    """
    he_init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.keras.layers.Dense(
        n, activation=activation, kernel_initializer=he_init, name="layer"
    )
    return layer(prev)
