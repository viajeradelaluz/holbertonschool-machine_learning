#!/usr/bin/env python3
"""
Module that updates the weights and biases of a neural network.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

    :param Y: one-hot np array (classes, m) that contains correct labels
        classes: number of classes
        m: number of data points
    :param weights: dictionary of weights and biases
    :param cache: dictionary of the outputs of each layer
    :param alpha: learning rate
    :param lambtha: L2 regularization parameter
    :param L: number of layers of the network

    - The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    - The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y

    for layer in range(L, 0, -1):
        A = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]
        b = weights["b" + str(layer)]

        dW = 1 / m * np.matmul(dz, A.T) + lambtha / m * W
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (1 - A**2)

        weights["W" + str(layer)] = W - alpha * dW
        weights["b" + str(layer)] = b - alpha * db
