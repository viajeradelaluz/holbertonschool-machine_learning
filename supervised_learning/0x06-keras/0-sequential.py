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
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)

    for layer in range(len(layers)):
        if layer == 0:
            model.add(
                K.layers.Dense(
                    layers[layer],
                    activation=activations[layer],
                    kernel_regularizer=L2,
                    input_shape=(nx,),
                )
            )
        else:
            model.add(K.layers.Dropout(rate=(1 - keep_prob)))
            model.add(
                K.layers.Dense(
                    layers[layer],
                    activation=activations[layer],
                    kernel_regularizer=L2,
                )
            )

    return model
