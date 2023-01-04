#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification.
"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter of the weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """Getter of the bias for the neuron"""
        return self.__b

    @property
    def A(self):
        """Getter of the predicted output of the neuron"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron."""
        gin = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-gin))  # sigmoid function
        return self.__A
