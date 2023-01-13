#!/usr/bin/env python3
"""
Module with task 1. Normalize
"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix.

    :param X: a np array of shape (d, nx) to normalize
        d: number of data points
        nx: number of features
    :param m: a np array of shape (nx,) with the mean of all features
    :param s: a np array of shape (nx,) with the standard deviation

    :return: normalized X matrix
    """
    return (X - m) / s
