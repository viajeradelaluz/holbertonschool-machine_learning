#!/usr/bin/env python3
"""
Module that saves and loads an entire model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves the model

    :param network: model to save
    :param filename: path of the file that the model should be saved to

    :return: None
    """
    network.save(filepath=filename)

    return None


def load_model(filename):
    """Loads an model

    :param filename: path of the file that the model should be loaded from

    :return: the loaded model
    """
    return K.models.load_model(filepath=filename)
