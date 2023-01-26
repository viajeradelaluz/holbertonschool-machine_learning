#!/usr/bin/env python3
"""
Module that saves and loads a model's configuration in JSON format.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves the model's configuration in JSON format

    :param network: model whose weights should be saved
    :param filename: path of the file that the weights should be saved to

    :return: None
    """
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)

    return None


def load_config(filename):
    """Loads the model with a specific configuration

    :param filename: path of the file that the weights should be saved to

    :return: the loaded model
    """
    with open(filename, "r") as f:
        config = f.read()

    return K.models.model_from_json(config)
