#!/usr/bin/env python3
"""
Module that builds a dense block as described in Densely Connected
Convolutional Networks.
"""

import tensorflow.keras as K


def dense_block(X, n_filters, growth_rate, layers):
    """Builds a dense block.

    :param X: the output from the previous layer.
    :param n_filters: an integer representing the number of filters in X.
    :param growth_rate: the growth rate for the dense block.
    :param layers: the number of layers in the dense block.

    - Use the bottleneck layers used for DenseNet-B
    - All weights use he normal initialization.
    - All convolutions are preceded by batch normalization and a rectified
    linear activation (ReLU), respectively.

    :return: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively.
    """

    # Convolution values: (filters, kernel_size, strides, padding)
    cvalues = ((4 * growth_rate, 1, 1, "same"), (growth_rate, 3, 1, "same"))

    for _ in range(layers):

        batch_1 = K.layers.BatchNormalization(axis=-1)(X)
        activation_1 = K.layers.Activation("relu")(batch_1)
        conv_1 = K.layers.Conv2D(
            *cvalues[0], activation="linear", kernel_initializer="he_normal"
        )(activation_1)

        batch_2 = K.layers.BatchNormalization(axis=-1)(conv_1)
        activation_2 = K.layers.Activation("relu")(batch_2)
        conv_2 = K.layers.Conv2D(
            *cvalues[1], activation="linear", kernel_initializer="he_normal"
        )(activation_2)

        X = K.layers.Concatenate(axis=-1)([X, conv_2])
        n_filters += growth_rate

    return X, n_filters
