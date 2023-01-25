#!/usr/bin/env python3
"""
Module that sets up Adam optimization for a keras model with categorical.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics.

    :param network: model to optimize
    :param alpha: learning rate
    :param beta1: first Adam optimization parameter
    :param beta2: second Adam optimization parameter

    :return: None
    """
    adam = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        loss="categorical_crossentropy",
        optimizer=adam,
        metrics=["accuracy"],
    )

    return None
