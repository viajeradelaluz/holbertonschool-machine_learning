#!/usr/bin/env python3
"""
Module that saves and loads a model's weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format="h5"):
    """Saves the model's weights

    :param network: model whose weights should be saved
    :param filename: path of the file that the weights should be saved to
    :param save_format: format in which the weights should be saved

    :return: None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Loads the model's weights

    :param network: model whose weights should be saved
    :param filename: path of the file that the weights should be saved to

    :return: None
    """
    network.load_weights(filepath=filename)
    return None
