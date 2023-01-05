#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification.
"""

import matplotlib.pyplot as plt
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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(self.__L):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")

            weight, bias = "W" + str(layer + 1), "b" + str(layer + 1)
            n = nx if layer == 0 else layers[layer - 1]
            std = np.sqrt(2 / n)

            # Weights initialized using the He et al. method
            self.__weights[weight] = np.random.randn(layers[layer], n) * std
            self.__weights[bias] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        """Getter method for the number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Getter method for the intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter method for the weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the deep neural network.
        :param X: input data.
        :return: the output of the neural network and the cache, respectively.
        """
        self.__cache["A0"] = X

        for layer in range(self.__L):
            z = (
                np.matmul(
                    self.__weights["W" + str(layer + 1)],
                    self.__cache["A" + str(layer)],
                )
                + self.__weights["b" + str(layer + 1)]
            )
            self.__cache["A" + str(layer + 1)] = 1 / (1 + np.exp(-z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression.
        :param Y: a np array with shape (1, m) with the correct labels
        :param A: a np array with shape (1, m) with the activated output
        :return: the cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions.
        :param X: a np array with shape (nx, m) with the input data
        :param Y: a np array with shape (1, m) with the correct labels
        :return: the neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent.
        :param Y: a np array with shape (1, m) with the correct labels
        :param cache: a dictionary containing all the intermediary values
        :param alpha: the learning rate
        :return: nothing
        """
        m = Y.shape[1]

        for layer in reversed(range(self.__L)):
            weight, bias = "W" + str(layer + 1), "b" + str(layer + 1)
            current, kache = "A" + str(layer), "A" + str(layer + 1)

            A = cache[kache]
            W = self.__weights[weight]

            mse = A - Y if layer == self.__L - 1 else H * (A * (1 - A))
            H = np.matmul(W.T, mse)

            dw = np.matmul(cache[current], mse.T) / m
            db = np.sum(mse, axis=1, keepdims=True) / m

            self.__weights[weight] -= alpha * dw.T
            self.__weights[bias] -= alpha * db

    def train(
        self,
        X,
        Y,
        iterations=5000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100,
    ):
        """Trains the neuron."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)

            if verbose and i % step == 0:
                costs.append(cost)
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(np.arange(0, iterations + 1), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()

        return self.evaluate(X, Y)
