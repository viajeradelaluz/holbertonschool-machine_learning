#!/usr/bin/env python3
"""
Module that builds the ResNet-50 architecture as described in Deep Residual
Learning for Image Recognition (2015).
"""

import tensorflow.keras as K

identity_block = __import__("2-identity_block").identity_block
projection_block = __import__("3-projection_block").projection_block


def resnet50():
    """Builds the ResNet-50 architecture.

    - Assumes the input data will have shape (224, 224, 3).
    - All convolutions are preceded by batch normalization along the channels
    axis and a rectified linear activation (ReLU), respectively.
    - All weights use he normal initialization.

    :return: the keras model.
    """

    X = K.Input(shape=(224, 224, 3))

    # Pool values: (pool_size, strides, padding)
    pvalues = ((3, 2, "same"), (7, 7, "valid"))

    conv_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        activation="linear",
        kernel_initializer="he_normal",
    )(X)
    batch_1 = K.layers.BatchNormalization(axis=-1)(conv_1)
    act_1 = K.layers.Activation("relu")(batch_1)
    pool_1 = K.layers.MaxPool2D(*pvalues[0])(act_1)

    filters = [64, 64, 256]
    conv_2_1 = projection_block(pool_1, filters, s=1)
    identity_2_1 = identity_block(conv_2_1, filters)
    identity_2_2 = identity_block(identity_2_1, filters)

    filters = [128, 128, 512]
    conv_3_1 = projection_block(identity_2_2, filters, s=2)
    identity_3_1 = identity_block(conv_3_1, filters)
    identity_3_2 = identity_block(identity_3_1, filters)
    identity_3_3 = identity_block(identity_3_2, filters)

    filters = [256, 256, 1024]
    conv_4_1 = projection_block(identity_3_3, filters, s=2)
    identity_4_1 = identity_block(conv_4_1, filters)
    identity_4_2 = identity_block(identity_4_1, filters)
    identity_4_3 = identity_block(identity_4_2, filters)
    identity_4_4 = identity_block(identity_4_3, filters)
    identity_4_5 = identity_block(identity_4_4, filters)

    filters = [512, 512, 2048]
    conv_5_1 = projection_block(identity_4_5, filters, s=2)
    identity_5_1 = identity_block(conv_5_1, filters)
    identity_5_2 = identity_block(identity_5_1, filters)

    avg_pool = K.layers.AveragePooling2D(*pvalues[1])(identity_5_2)
    Y = K.layers.Dense(
        units=1000, activation="softmax", kernel_initializer="he_normal"
    )(avg_pool)

    model = K.Model(inputs=X, outputs=Y)

    return model
