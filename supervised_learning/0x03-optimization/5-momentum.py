#!/usr/bin/env python3
"""
Module with task 5. Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent with momentum.

    :param alpha: learning rate
    :param beta1: momentum weight
    :param var: a np array with the variable to update
    :param grad: a np array with the gradient of var
    :param v: the previous first moment of var

    :return: the updated variable and the new moment, respectively
    """
    momentum = beta1 * v + (1 - beta1) * grad
    updated_var = var - alpha * momentum

    return updated_var, momentum
