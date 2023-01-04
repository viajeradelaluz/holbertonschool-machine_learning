#!/usr/bin/env python3
"""
Module that defines a neural network with one hidden layer performing binary
classification.
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer."""

    def __init__(self, nx, nodes):
        """Class constructor.
        :param nx: number of input features.
        :param nodes: number of nodes found in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1, self.__b2 = np.zeros((nodes, 1)), 0
        self.__A1, self.__A2 = 0, 0

    @property
    def W1(self):
        """Getter for the weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter for the bias for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter for the weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output for the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network.
        :param X: a np array with shape (nx, m) that contains the input data.
        :return: the private attributes __A1 and __A2, respectively.
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression.
        :param Y: a np array with shape (1, m) with the correct labels
        :param A: a np array with shape (1, m) with the activated output
        :return: the cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
