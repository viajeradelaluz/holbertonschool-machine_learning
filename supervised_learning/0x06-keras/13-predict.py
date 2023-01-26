#!/usr/bin/env python3
"""
Module that makes a prediction using a neural network.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network

    :param network: model to test
    :param data: input data to test the model with
    :param verbose: bool, determines if output is printed during prediction

    :return: the prediction for the data
    """
    return network.predict(x=data, verbose=verbose)
