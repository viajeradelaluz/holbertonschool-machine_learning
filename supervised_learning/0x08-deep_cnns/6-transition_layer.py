#!/usr/bin/env python3
"""
Module that builds a transition layer as described in Densely Connected
Convolutional Networks.
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer.

    :param X: the output from the previous layer.
    :param nb_filters: an integer representing the number of filters in X.
    :param compression: the compression factor for the transition layer.

    - Implement compression as used in DenseNet-C
    - All weights use he normal initialization.
    - All convolutions are preceded by batch normalization and a rectified
    linear activation (ReLU), respectively.

    :return: The output of the transition layer and the number of filters
    within the output, respectively.
    """

    n_filters = int(nb_filters * compression)
    batch = K.layers.BatchNormalization(axis=-1)(X)
    activation = K.layers.Activation("relu")(batch)

    # Convolution values: (filters, kernel_size, strides, padding)
    cvalues = (n_filters, 1, 1, "same")

    conv = K.layers.Conv2D(
        *cvalues, activation="linear", kernel_initializer="he_normal"
    )(activation)

    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)

    return avg_pool, n_filters
