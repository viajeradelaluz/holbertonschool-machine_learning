#!/usr/bin/env python3
"""
Module with task 0. Normalization Constants
"""

import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix:

    :param X: is the numpy.ndarray of shape (m, nx) to normalize
    - m: is the number of data points
    - nx: is the number of features

    :return: the mean and std deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
