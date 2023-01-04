#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network."""

    def __init__(self, nx, layers):
        """Class constructor.
        :param nx: number of input features.
        :param layers: list representing the number of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer in range(self.L):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")

            weight, bias = "W" + str(layer + 1), "b" + str(layer + 1)
            n = nx if layer == 0 else layers[layer - 1]
            std = np.sqrt(2 / n)

            # Weights initialized using the He et al. method
            self.weights[weight] = np.random.randn(layers[layer], n) * std
            self.weights[bias] = np.zeros((layers[layer], 1))
