#!/usr/bin/env python3
"""
Module that builds the DenseNet-121 architecture as described in Densely
Connected Convolutional Networks.
"""
import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture.

    :param growth_rate: the growth rate.
    :param compression: the compression factor.

    - The input data will have shape (224, 224, 3).
    - All convolutions are preceded by batch normalization and a rectified
    linear activation (ReLU), respectively.
    - All weights use he normal initialization.

    :return: the keras model.
    """

    X = K.Input(shape=(224, 224, 3))
    batch = K.layers.BatchNormalization(axis=-1)(X)
    activation = K.layers.Activation("relu")(batch)

    # Convolution values: (filters, kernel_size, strides, padding)
    cvalues = (2 * growth_rate, 7, 2, "same")

    conv = K.layers.Conv2D(
        *cvalues, activation="linear", kernel_initializer="he_normal"
    )(activation)

    pool = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(conv)

    block_1, filters = dense_block(pool, pool.shape[-1], growth_rate, layers=6)
    tlayer_1, filters = transition_layer(block_1, filters, compression)

    block_2, filters = dense_block(tlayer_1, filters, growth_rate, layers=12)
    tlayer_2, filters = transition_layer(block_2, filters, compression)

    block_3, filters = dense_block(tlayer_2, filters, growth_rate, layers=24)
    tlayer_3, filters = transition_layer(block_3, filters, compression)

    block_4, filters = dense_block(tlayer_3, filters, growth_rate, layers=16)

    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=7)(block_4)
    Y = K.layers.Dense(
        units=1000, activation="softmax", kernel_initializer="he_normal"
    )(avg_pool)

    model = K.Model(inputs=X, outputs=Y)

    return model
