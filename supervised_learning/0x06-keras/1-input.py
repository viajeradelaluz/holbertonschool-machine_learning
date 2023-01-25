#!/usr/bin/env python3
"""
Module that builds a neural network with the Keras library.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library

    :param nx: number of input features to the network
    :param layers: list with the number of nodes in each layer of the network
    :param activations: list with the activation functions used for each layer
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout

    :return: the keras model
    """
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    rate = 1 - keep_prob
    d = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=L2,
    )
    outputs = d(inputs)

    for layer in range(1, len(layers)):
        outputs = K.layers.Dropout(rate=rate)(outputs)
        d = K.layers.Dense(
            layers[layer],
            activation=activations[layer],
            kernel_regularizer=L2,
        )
        outputs = d(outputs)

    model = K.Model(inputs=inputs, outputs=outputs)

    return model
