#!/usr/bin/env python3
"""
Module that builds an inception block as described in Going Deeper with
Convolutions (2014).
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block.

    :param A_prev: is the output from the previous layer.
    :param filters: is a tuple or list containing F1, F3R, F3, F5R, F5, FPP:
        - F1: number of filters in the 1x1 convolution.
        - F3R: number of filters in the 1x1 convolution before the 3x3 conv.
        - F3: number of filters in the 3x3 convolution.
        - F5R: number of filters in the 1x1 convolution before the 5x5 conv.
        - F5: number of filters in the 5x5 convolution.
        - FPP: number of filters in the 1x1 convolution after the max pooling.

    - All convolutions inside the inception block should use a rectified
    linear activation (ReLU).

    :return: the concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_F1 = K.layers.Conv2D(
        filters=F1, kernel_size=1, padding="same", activation="relu"
    )(A_prev)

    conv_F3R = K.layers.Conv2D(
        filters=F3R, kernel_size=1, padding="same", activation="relu"
    )(A_prev)

    conv_F3 = K.layers.Conv2D(
        filters=F3, kernel_size=3, padding="same", activation="relu"
    )(conv_F3R)

    conv_F5R = K.layers.Conv2D(
        filters=F5R, kernel_size=1, padding="same", activation="relu"
    )(A_prev)

    conv_F5 = K.layers.Conv2D(
        filters=F5, kernel_size=5, padding="same", activation="relu"
    )(conv_F5R)

    pool = K.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(A_prev)

    conv_FPP = K.layers.Conv2D(
        filters=FPP, kernel_size=1, padding="same", activation="relu"
    )(pool)

    return K.layers.Concatenate(axis=-1)([conv_F1, conv_F3, conv_F5, conv_FPP])
