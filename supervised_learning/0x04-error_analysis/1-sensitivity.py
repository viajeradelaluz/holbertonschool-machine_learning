#!/usr/bin/env python3
"""
Module that calculates the sensitivity for each class in a confusion matrix.
"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix.

    :param confusion: a np array (classes, classes) where row indices are the
    correct labels and column indices are the predicted labels
        classes: number of classes

    :return: a np array (classes,) containing the sensitivity of each class
    """

    true_positive = np.diagonal(confusion)
    positive = np.sum(confusion, axis=1)

    return true_positive / positive
