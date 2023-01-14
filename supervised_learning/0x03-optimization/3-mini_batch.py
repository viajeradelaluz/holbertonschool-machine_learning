#!/usr/bin/env python3
"""
Module with task 3. Mini-batch
"""

import tensorflow.compat.v1 as tf

shuffle_data = __import__("2-shuffle_data").shuffle_data


def train_mini_batch(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    batch_size=32,
    epochs=5,
    load_path="/tmp/model.ckpt",
    save_path="/tmp/model.ckpt",
):
    """Trains a neural network model using mini-batch gradient descent.

    :param X_train: a np array of shape (m, 784) with the training data
    :param Y_train: a np array of shape (m, 10) with the training labels
    :param X_valid: a np array of shape (m, 784) with the validation data
    :param Y_valid: a np array of shape (m, 10) with the validation labels
    :param batch_size: the number of data points in a batch
    :param epochs: number of iterations to train over
    :param load_path: path from which to load the model
    :param save_path: path to save the model

    :return: the path where the model was saved
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        if m % batch_size == 0:
            mini_batch = m // batch_size
        else:
            mini_batch = m // batch_size + 1

        for i in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train}
            )
            validation_cost, validation_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(validation_cost))
            print("\tValidation Accuracy: {}".format(validation_accuracy))

            if i < epochs:
                x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)

                for j in range(mini_batch):
                    start = j * batch_size
                    step = j + 1
                    end = start + batch_size

                    x_mini = x_shuffle[start:end]
                    y_mini = y_shuffle[start:end]

                    sess.run(train_op, feed_dict={x: x_mini, y: y_mini})

                    if step % 100 == 0:
                        cost_mini, accuracy_mini = sess.run(
                            [loss, accuracy], feed_dict={x: x_mini, y: y_mini}
                        )
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(cost_mini))
                        print("\t\tAccuracy: {}".format(accuracy_mini))

        return saver.save(sess, save_path)
