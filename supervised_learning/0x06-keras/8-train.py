#!/usr/bin/env python3
"""
Module that trains a model and saves the best iteration.
"""
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    early_stopping=False,
    patience=0,
    learning_rate_decay=False,
    alpha=0.1,
    decay_rate=1,
    save_best=False,
    filepath=None,
    verbose=True,
    shuffle=False,
):
    """Trains a model and saves the best iteration.

    :param network: model to train
    :param data: np array (m, nx) with the input data
    :param labels: one-hot np array (m, classes) with the labels of data
    :param batch_size: batch size used for mini-batch gradient descent
    :param epochs: number of iterations to train over
    :param validation_data: data to validate the model with, if not None
    :param early_stopping: bool, determines if early stopping should be used
    :param patience: patience used for early stopping
    :param learning_rate_decay: bool, determines if learning rate decay is used
    :param alpha: initial learning rate
    :param decay_rate: decay rate
    :param save_best: bool, determines if the iteration should be saved
    :param filepath: path to save the model
    :param verbose: bool, determines if output is printed during training
    :param shuffle: bool, determines if the batches are shuffled every epoch

    :return: the History object generated after training the model
    """

    def scheduler(epoch):
        """Updates the learning rate using inverse time decay"""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if validation_data:
        if early_stopping:
            callbacks = K.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
            )

        if learning_rate_decay:
            callbacks = K.callbacks.LearningRateScheduler(
                schedule=scheduler,
                verbose=1,
            )
        if save_best:
            callbacks.append(
                K.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    save_best_only=save_best,
                )
            )

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        shuffle=shuffle,
    )

    return history
