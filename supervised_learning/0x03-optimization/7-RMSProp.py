#!/usr/bin/env python3
"""
Module with task 7. RMSProp
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm.

    :param alpha: learning rate
    :param beta2: RMSProp weight
    :param epsilon: small number to avoid division by zero
    :param var: a np array with the variable to update
    :param grad: a np array with the gradient of var
    :param s: the previous second moment of var

    :return: the updated variable and the new moment, respectively
    """
    sdw = beta2 * s + (1 - beta2) * grad**2
    updated_var = var - alpha * grad / (sdw ** (1 / 2) + epsilon)

    return updated_var, sdw
