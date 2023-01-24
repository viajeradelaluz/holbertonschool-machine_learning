#!/usr/bin/env python3
"""
Module that conducts forward propagation using Dropout.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    :param X: np darray (nx, m) containing the input data
        nx: number of input features
        m: number of data points
    :param weights: dictionary of the weights and biases
    :param L: number of layers
    :param keep_prob: probability that a node will be kept

    :return: a dictionary with the outputs of each layer and the dropout mask
    used on each layer.
    """

    cache = {"A0": X}

    for layer in range(L):
        W = weights["W" + str(layer + 1)]
        b = weights["b" + str(layer + 1)]
        A = cache["A" + str(layer)]
        Z = np.matmul(W, A) + b

        if layer == L - 1:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A *= D
            A /= keep_prob
            cache["D" + str(layer + 1)] = D

        cache["A" + str(layer + 1)] = A

    return cache
