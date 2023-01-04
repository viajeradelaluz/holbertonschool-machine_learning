#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification.
"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor.
        :param nx: number of input features to the neuron
        """
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
        """Calculates the forward propagation of the neuron.
        :param X: a numpy.ndarray with shape (nx, m) with the input data
        :return: the private attribute __A
        """
        gin = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-gin))  # sigmoid function
        return self.__A

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
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient on the neuron.
        :param X: a np array with shape (nx, m) with the input data
        :param Y: a np array with shape (1, m) with the correct labels
        :param A: a np array with shape (1, m) with the activated output
        :param alpha: the learning rate
        :return: nothing
        """
        m = Y.shape[1]
        mse = A - Y  # derivative of the cost function
        dw = np.matmul(X, mse.T) / m  # gradient of the weights
        db = np.sum(mse) / m  # gradient of the bias
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db
