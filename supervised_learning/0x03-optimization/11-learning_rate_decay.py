#!/usr/bin/env python3
"""
Module with task 11. Learning Rate Decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy.

    :param alpha: the original learning rate
    :param decay_rate: weight to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have
    elapsed
    :param decay_step: the number of passes of gradient descent that should
    occur before alpha is decayed further

    :return: the updated value for alpha
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
