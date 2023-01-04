#!/usr/bin/env python3
"""
Module that defines a neural network with one hidden layer performing binary
classification.
"""

import matplotlib.pyplot as plt
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

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions.
        :param X: a np array with shape (nx, m) with the input data
        :param Y: a np array with shape (1, m) with the correct labels
        :return: the neuron's prediction and the cost of the network
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient on the neuron.
        :param X: a np array with shape (nx, m) with the input data
        :param Y: a np array with shape (1, m) with the correct labels
        :param A1: the output of the hidden layer
        :param A2: the predicted output
        :param alpha: the learning rate
        :return: nothing
        """
        m = X.shape[1]

        # Gradient descent of hidden layer
        mse2 = A2 - Y
        dw2 = np.matmul(mse2, A1.T) / m
        db2 = np.sum(mse2, axis=1, keepdims=True) / m

        # Gradient descent of output layer
        mse1 = np.matmul(self.__W2.T, mse2) * (A1 * (1 - A1))  # sigmoid
        dw1 = np.matmul(mse1, X.T) / m
        db1 = np.sum(mse1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron.
        :param X: a np array with shape (nx, m) with the input data
        :param Y: a np array with shape (1, m) with the correct labels
        :param iterations: the number of iterations to train over
        :param alpha: the learning rate
        :return: the evaluation of the training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
