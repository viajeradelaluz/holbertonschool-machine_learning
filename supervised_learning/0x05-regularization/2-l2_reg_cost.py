#!/usr/bin/env python3
"""
Module that calculates the cost of a neural network with L2 regularization.
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization

    :param cost: a tensor containing the cost without L2 reg

    :return: a tensor containing the cost with L2 regularization
    """

    return cost + tf.losses.get_regularization_losses()
