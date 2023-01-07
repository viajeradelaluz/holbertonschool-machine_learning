#!/usr/bin/env python3
"""
Module of the function create_placeholders.
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Create placeholders, x and y, for the neural network.
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return: placeholders named x and y, respectively
    """
    x = tf.placeholder(float, shape=(None, nx), name="x")
    y = tf.placeholder(float, shape=(None, classes), name="y")

    return x, y
