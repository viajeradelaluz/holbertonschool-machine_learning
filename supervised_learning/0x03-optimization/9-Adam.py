#!/usr/bin/env python3
"""
Module with task 9. Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm.

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
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad**2

    vdw_corrected = vdw / (1 - beta1**t)
    sdw_corrected = sdw / (1 - beta2**t)

    var = var - alpha * (vdw_corrected / (np.sqrt(sdw_corrected) + epsilon))

    return var, vdw, sdw
