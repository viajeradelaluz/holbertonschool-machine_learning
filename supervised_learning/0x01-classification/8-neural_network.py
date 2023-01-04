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

        self.W1 = np.random.randn(nodes, nx)
        self.W2 = np.random.randn(1, nodes)
        self.b1, self.b2 = np.zeros((nodes, 1)), 0
        self.A1, self.A2 = 0, 0
