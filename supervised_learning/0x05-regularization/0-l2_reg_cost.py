#!/usr/bin/env python3
"""
Module that calculates the cost of a neural network with L2 regularization.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    :param cost: cost of the network without L2 regularization
    :param lambtha: regularization parameter
    :param weights: dictionary of the weights and biases of the neural network
    :param L: number of layers in the neural network
    :param m: number of data points used

    :return: the cost of the network accounting for L2 regularization
    """

    L2 = 0
    for layer in range(L):
        zigma = np.linalg.norm(weights["W" + str(layer + 1)])
        L2 += lambtha / 2 / m * zigma**2

    return L2 + cost
