#!/usr/bin/env python3
"""
Class that represents a poisson distribution.
"""


class Poisson:
    """Poisson distribution"""

    def __init__(self, data=None, lambtha=1.0):
        """Poisson distribution constructor

        :param data: list of the data to be used to estimate the distribution
        :param lambtha: expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF for a given number of successes."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # Ref. Distribución de Poisson PMF: https://youtu.be/PMX75m4-s9A
        e = 2.7182818285
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i
        return (e**-self.lambtha) * (self.lambtha**k) / k_factorial

    def cdf(self, k):
        """Calculates the CDF for a given number of successes."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        # Ref. Distribución de Poisson CDF: https://youtu.be/x9jF11I5x-g
        e = 2.7182818285
        cdf = [self.pmf(i) for i in range(k + 1)]
        return sum(cdf)
