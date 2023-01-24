#!/usr/bin/env python3
"""
Module that calculates the precision for each class in a confusion matrix.
"""

import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix.

    :param confusion: a np array (classes, classes) where row indices are the
    correct labels and column indices are the predicted labels
        classes: number of classes

    :return: a np array (classes,) containing the precision of each class
    """

    true_positive = np.diagonal(confusion)
    true_plus_false_positive = np.sum(confusion, axis=0)

    return true_positive / true_plus_false_positive
