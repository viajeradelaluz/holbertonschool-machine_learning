#!/usr/bin/env python3
"""
Module to calculate the softmax cross-entropy loss of a prediction
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction
    :param y: a placeholders with the right labels of the input data
    :param y_pred: tensor containing the network's predictions
    :return: a tensor containing the loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
