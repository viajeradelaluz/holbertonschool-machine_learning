#!/usr/bin/env python3
"""
Module that trains a model using mini-batch gradient descent to also analyze
validation data.
"""
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    verbose=True,
    shuffle=False,
):
    """Train a model to also analyze validation data.

    :param network: model to train
    :param data: np array (m, nx) with the input data
    :param labels: one-hot np array (m, classes) with the labels of data
    :param batch_size: batch size used for mini-batch gradient descent
    :param epochs: number of iterations to train over
    :param validation_data: data to validate the model with, if not None
    :param verbose: bool, determines if output is printed during training
    :param shuffle: bool, determines if the batches are shuffled every epoch

    :return: the History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
    )
    return history
