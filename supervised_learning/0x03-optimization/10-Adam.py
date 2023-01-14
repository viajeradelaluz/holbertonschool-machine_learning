#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm with tensorflow
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm.

    :param alpha: the learning rate
    :param beta1: the weight used for the first moment
    :param beta2: the weight used for the second moment
    :param epsilon: small number to avoid division by zero
    :param var: a np array with the variable to be updated
    :param grad: a np array with the gradient of var
    :param v: the previous first moment of var
    :param s: the previous second moment of var
    :param t: the time step used for bias correction

    :return: the updated variable, the new first moment, and the new second
    moment, respectively
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
