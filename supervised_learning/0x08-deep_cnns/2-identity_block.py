#!/usr/bin/env python3
"""
Module that builds an identity block as described in Deep Residual Learning
for Image Recognition (2015).
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block.

    :param A_prev: output from the previous layer.
    :param filters: tuple or list containing F11, F3, F12, respectively:
        - F11: number of filters in the first 1x1 convolution.
        - F3: number of filters in the 3x3 convolution.
        - F12: number of filters in the second 1x1 convolution.

    - All convolutions follow a batch normalization along the channels axis
    and a rectified linear activation.
    - All weights use he normal initialization.

    :return: the activated output of the identity block.
    """

    F11, F3, F12 = filters

    # Convolution values: (filters, kernel_size, strides, padding)
    cvalues = ((F11, 1, 1, "same"), (F3, 3, 1, "same"), (F12, 1, 1, "same"))

    conv_F11 = K.layers.Conv2D(
        *cvalues[0], activation="linear", kernel_initializer="he_normal"
    )(A_prev)
    batch_F11 = K.layers.BatchNormalization(axis=-1)(conv_F11)
    activation_F11 = K.layers.Activation("relu")(batch_F11)

    conv_F3 = K.layers.Conv2D(
        *cvalues[1], activation="linear", kernel_initializer="he_normal"
    )(activation_F11)
    batch_F3 = K.layers.BatchNormalization(axis=-1)(conv_F3)
    activation_F3 = K.layers.Activation("relu")(batch_F3)

    conv_F12 = K.layers.Conv2D(
        *cvalues[2], activation="linear", kernel_initializer="he_normal"
    )(activation_F3)
    batch_F12 = K.layers.BatchNormalization(axis=-1)(conv_F12)

    X = K.layers.Add()([batch_F12, A_prev])
    activated_output = K.layers.Activation("relu")(X)

    return activated_output
