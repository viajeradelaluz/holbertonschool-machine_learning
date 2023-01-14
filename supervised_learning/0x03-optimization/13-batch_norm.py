#!/usr/bin/env python3
"""
Module with task 13. Batch Normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
    normalization.

    :param Z: a np array of shape (m, n) that should be normalized
    :param gamma: a np array of shape (1, n) with the scales used
    for batch normalization
    :param beta: a np array of shape (1, n) with the offsets used
    for batch normalization
    :param epsilon: a small number used to avoid division by zero

    :return: the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * normalized + beta
