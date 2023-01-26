#!/usr/bin/env python3
"""
Module that tests a neural network.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network

    :param network: model to test
    :param data: input data to test the model with
    :param labels: correct one-hot labels of data
    :param verbose: bool, determines if output should be printed during testing

    :return: the loss and accuracy with the testing data, respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
