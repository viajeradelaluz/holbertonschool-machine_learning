#!/usr/bin/env python3
"""
Module that converts a one-hot matrix into a vector of labels.
"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels.
    :param one_hot: a one-hot encoded np array with shape (classes, m)
    return: a np array with shape (m,) or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
