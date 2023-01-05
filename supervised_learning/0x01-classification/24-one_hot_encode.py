#!/usr/bin/env python3
"""
Module that converts a numeric label vector into a one-hot matrix
"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix.
    :param Y: a np array with shape (m,) containing numeric class labels
    :param classes: the maximum number of classes found in Y
    return: a one-hot of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray):
        return None

    try:
        one_hot = np.eye(classes)[Y]
        return one_hot.T
    except Exception:
        return None
