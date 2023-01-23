#!/usr/bin/env python3
"""
Module that creates a confusion matrix.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    :param labels: one-hot np array (m, classes) with the correct labels:
        m: number of data points
        classes: number of classes
    :param logits: one-hot np array (m, classes) with the predicted labels.

    :return: a confusion np array (classes, classes) with row indices with the
    correct labels and column indices with predicted labels
    """

    return np.matmul(labels.T, logits)
