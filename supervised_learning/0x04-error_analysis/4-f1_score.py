#!/usr/bin/env python3
"""
Module that calculates the F1 score of a confusion matrix.
"""

import numpy as np

sensitivity = __import__("1-sensitivity").sensitivity
precision = __import__("2-precision").precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix

    :param confusion: a np array (classes, classes) where row indices are the
    correct labels and column indices are the predicted labels
        classes: number of classes

    :return: a np array (classes,) containing the specificity of each class
    """

    recall = sensitivity(confusion)
    precision_ = precision(confusion)

    return 2 * (recall * precision_) / (recall + precision_)
