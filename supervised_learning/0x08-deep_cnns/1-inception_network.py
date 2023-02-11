#!/usr/bin/env python3
"""
Module that builds the inception network as described in Going Deeper with
Convolutions (2014).
"""

import tensorflow.keras as K

inception_block = __import__("0-inception_block").inception_block


def inception_network():
    """Builds the inception network.

    - Input data will have shape (224, 224, 3)
    - All convolutions inside and outside the inception block should use a
    rectified linear activation (ReLU)
    - Use inception_block = __import__('0-inception_block').inception_block

    :return: the keras model
    """

    X = K.Input(shape=(224, 224, 3))
    convolution, max_pool = K.layers.Conv2D, K.layers.MaxPool2D
    inception = inception_block

    # Convolution values: (filters, kernel_size, strides, padding)
    cvalues = ((64, 7, 2, "same"), (192, 3, 1, "same"))

    # Pool values: (pool_size, strides, padding)
    pvalues = ((3, 2, "same"), (7, 1, "valid"))

    conv_1 = convolution(*cvalues[0], activation="relu")(X)
    pool_1 = max_pool(*pvalues[0])(conv_1)

    conv_2 = convolution(*cvalues[1], activation="relu")(pool_1)
    pool_2 = max_pool(*pvalues[0])(conv_2)

    incept_3a = inception(pool_2, filters=[64, 96, 128, 16, 32, 32])
    incept_3b = inception(incept_3a, filters=[128, 128, 192, 32, 96, 64])
    pool_3 = max_pool(*pvalues[0])(incept_3b)

    incept_4a = inception(pool_3, filters=[192, 96, 208, 16, 48, 64])
    incept_4b = inception(incept_4a, filters=[160, 112, 224, 24, 64, 64])
    incept_4c = inception(incept_4b, filters=[128, 128, 256, 24, 64, 64])
    incept_4d = inception(incept_4c, filters=[112, 144, 288, 32, 64, 64])
    incept_4e = inception(incept_4d, filters=[256, 160, 320, 32, 128, 128])
    pool4 = max_pool(*pvalues[0])(incept_4e)

    incept_5a = inception(pool4, filters=[256, 160, 320, 32, 128, 128])
    incept_5b = inception(incept_5a, filters=[384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(*pvalues[1])(incept_5b)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    Y = K.layers.Dense(units=1000, activation="softmax")(dropout)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
