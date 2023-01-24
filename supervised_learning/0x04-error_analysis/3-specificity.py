#!/usr/bin/env python3
"""
Module that calculates the specificity for each class in a confusion matrix.
"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    :param confusion: a np array (classes, classes) where row indices are the
    correct labels and column indices are the predicted labels
        classes: number of classes

    :return: a np array (classes,) containing the specificity of each class
    """

    # False Negative (FN), False Positive (FP)
    # True Negative (TN), True Positive (TP)

    TP = np.diagonal(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP

    TN = np.sum(confusion) - TP - FN - FP

    return TN / (TN + FP)
