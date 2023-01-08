#!/usr/bin/env python3
"""
Module that evaluates the output of a neural network.
"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network.
    :param X: a np darray containing the input data to evaluate
    :param Y: a np darray containing the one-hot labels for X
    :param save_path: is the location to load the model from
    :return: the network's prediction, accuracy, and loss, respectively
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        y_pred, accuracy, loss = (
            tf.get_collection("y_pred")[0],
            tf.get_collection("accuracy")[0],
            tf.get_collection("loss")[0],
        )

        y_val, accuracy_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y}
        )

        return y_val, accuracy_val, loss_val
