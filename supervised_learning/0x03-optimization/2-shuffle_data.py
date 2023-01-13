#!/usr/bin/env python3
"""
Module with task 2. Shuffle data
"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    :param X: a np array of shape (m, nx) to shuffle
        m: number of data points
        nx: number of features in X
    :param Y: a np array of shape (m, ny) to shuffle
        m: number of data points
        ny: number of features in Y

    :return: shuffled X and Y matrices.
    """
    permuted = np.random.permutation(X.shape[0])

    return X[permuted], Y[permuted]
