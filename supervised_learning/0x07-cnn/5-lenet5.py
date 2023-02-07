#!/usr/bin/env python3
"""
Module that builds a modified version of the LeNet-5 architecture using keras.
"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using keras.

    :param X: a K.Input of shape (m, 28, 28, 1) with the input images.
        m: the number of images.

    - The model consists of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding.
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding.
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        - Fully connected layer with 120 nodes.
        - Fully connected layer with 84 nodes.
        - Fully connected softmax output layer with 10 nodes.
    - All layers are initialized using he_normal initialization method.

    :return: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics.
    """

    init = K.initializers.he_normal(seed=None)

    layer_1 = K.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=init,
    )(X)
    pool1 = K.layers.MaxPool2D(pool_size=2, strides=2)(layer_1)

    layer_2 = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        kernel_initializer=init,
        activation="relu",
    )(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=2, strides=2)(layer_2)

    flatten = K.layers.Flatten()(pool2)

    connected_layer_1 = K.layers.Dense(
        units=120, activation="relu", kernel_initializer=init
    )(flatten)

    connected_layer_2 = K.layers.Dense(
        units=84, activation="relu", kernel_initializer=init
    )(connected_layer_1)

    connected_layer_3 = K.layers.Dense(
        units=10, activation="softmax", kernel_initializer=init
    )(connected_layer_2)

    model = K.Model(inputs=X, outputs=connected_layer_3)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
