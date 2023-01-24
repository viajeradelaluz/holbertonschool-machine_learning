#!/usr/bin/env python3
"""
Module that updates the weights and biases of a neural network.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

    :param Y: one-hot np array (classes, m) with the correct labels
        classes: number of classes
        m: number of data points
    :param weights: dictionary of weights and biases
    :param cache: dictionary of the outputs of each layer
    :param alpha: learning rate
    :param keep_prob: probability that a node will be kept
    :param L: number of layers of the network

    - All layers except the last use the tanh activation function, softmax
    activation function on the last layer
    - The weights and biases of the network should be updated in place
    """

    m = Y.shape[1]

    for layer in reversed(range(L)):
        A = cache["A" + str(layer + 1)]
        A_prev = cache["A" + str(layer)]

        if layer == L - 1:
            dZ = A - Y
        else:
            dZ = da * (1 - A**2)
            dZ *= cache["D" + str(layer + 1)]
            dZ /= keep_prob

        W = weights["W" + str(layer + 1)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        da = np.matmul(W.T, dZ)

        weights["W" + str(layer + 1)] -= alpha * dW
        weights["b" + str(layer + 1)] -= alpha * db
