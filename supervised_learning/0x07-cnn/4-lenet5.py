#!/usr/bin/env python3
"""
Module that builds a modified version of the LeNet-5 architecture.
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture using tensorflow.

    :param x: a tf.placeholder (m, 28, 28, 1) with the input images.
        m: the number of images.
    :param y: a tf.placeholder (m, 10) with the one-hot labels for the images.

    - The model consists of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding.
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding.
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides.
        - Fully connected layer with 120 nodes.
        - Fully connected layer with 84 nodes.
        - Fully connected softmax output layer with 10 nodes.
    - All layers are initialized using he_normal initialization method:
        - tf.keras.initializers.VarianceScaling(scale=2.0)

    :return:
        - a tensor for the softmax activated output
        - a training operation with Adam optimization (default hyperparameters)
        - a tensor for the loss of the network
        - a tensor for the accuracy of the network
    """

    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    layer_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer=init,
    )(x)
    pool_1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer_1)

    layer_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        kernel_initializer=init,
        activation="relu",
    )(pool_1)
    pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer_2)

    flatten = tf.layers.Flatten()(pool2)

    connected_layer_1 = tf.layers.Dense(
        units=120, activation="relu", kernel_initializer=init
    )(flatten)
    
    connected_layer_2 = tf.layers.Dense(
        units=84, activation="relu", kernel_initializer=init
    )(connected_layer_1)
    
    connected_layer_3 = tf.layers.Dense(
        units=10, activation=None, kernel_initializer=init
    )(connected_layer_2)

    pred = tf.equal(tf.argmax(y, 1), tf.argmax(connected_layer_3, 1))

    softmax = tf.nn.softmax(connected_layer_3)
    loss = tf.losses.softmax_cross_entropy(y, logits=connected_layer_3)
    adam = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    return softmax, adam, loss, accuracy
