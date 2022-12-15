#!/usr/bin/env python3
"""
Class that represents a normal distribution.
"""


from distutils.command import sdist


class Normal:
    """Normal distribution class."""

    def __init__(self, data=None, mean=0.0, stddev=1.0):
        """Normal distribution class constructor.

        :param data: list of the data to be used to estimate the distribution.
        :param mean: mean of the distribution.
        :param stddev: standard deviation of the distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sd = sum([(x - self.mean) ** 2 for x in data]) / len(data)
            self.stddev = sd ** 0.5
