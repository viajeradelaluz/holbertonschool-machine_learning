#!/usr/bin/env python3
"""
Class that represents a normal distribution.
"""


class Normal:
    """Normal distribution class."""

    e = 2.7182818285
    pi = 3.1415926536

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
            self.stddev = sd**0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the PDF value for a given x-value."""

        # Calculation with standard deviation and mean
        dy = self.e ** (-((x - self.mean) ** 2) / (2 * self.stddev**2))
        dx = self.stddev * ((2 * self.pi) ** 0.5)
        return dy / dx
