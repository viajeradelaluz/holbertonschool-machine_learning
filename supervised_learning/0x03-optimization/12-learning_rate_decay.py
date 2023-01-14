#!/usr/bin/env python3
"""
Module with task 12. Learning Rate Decay Upgraded
"""

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation in tensorflow using inverse
    time decay.

    :param alpha: the original learning rate
    :param decay_rate: weight to determine the rate at which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: the number of passes of gradient descent that should
    occur before alpha is decayed further

    :return: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )
