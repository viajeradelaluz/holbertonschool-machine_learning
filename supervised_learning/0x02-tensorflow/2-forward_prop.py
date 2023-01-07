#!/usr/bin/env python3
"""
Module that creates the forward propagation graph for the neural network
"""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.
    :param x: the placeholder for the input data
    :param layer_sizes: the number of nodes in each layer
    :param activation: the activation functions for each layer
    :return: the prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            predict = create_layer(x, layer_sizes[0], activations[0])
        else:
            predict = create_layer(predict, layer_sizes[i], activations[i])
    return predict
