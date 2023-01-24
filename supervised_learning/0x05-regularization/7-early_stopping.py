#!/usr/bin/env python3
"""
Module that determines if you should stop gradient descent early.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early.

    :param cost: current validation cost of the neural network
    :param opt_cost: lowest recorded validation cost of the neural network
    :param threshold: threshold used for early stopping
    :param patience: patience count used for early stopping
    :param count: count of how long the threshold has not been met

    :return: a boolean of whether the network should be stopped early,
    followed by the updated count
    """

    count = 0 if opt_cost - cost > threshold else count + 1

    if count == patience:
        return True, count

    return False, count
