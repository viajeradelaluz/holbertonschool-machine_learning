#!/usr/bin/env python3
"""
Module with task 4. Moving Average
"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set.

    :param data: list of data to calculate the moving average of
    :param beta: weight used for the moving average

    :return: list containing the moving averages of data
    """
    vt = 0
    moving_average = []

    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        bias_correction = 1 - beta ** (i + 1)
        moving_average.append(vt / bias_correction)

    return moving_average
