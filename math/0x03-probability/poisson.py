#!/usr/bin/env python3
"""
Create a class Poisson that represents a poisson distribution:
- Class contructor def __init__(self, data=None, lambtha=1.):
  - `data` is a list of the data to be used to estimate the distribution
  - `lambtha` is the expected number of occurences in a given time frame
  - Sets the instance attribute `lambtha`
    - Saves `lambtha` as a float
  - If `data` is not given, i.e. None (be careful: `not data` has not the same
  result as `data is None`):
    - Use the given `lambtha`
    - If `lambtha` is not a positive value or equals to 0, raise a `ValueError`
    with the message `lambtha must be a positive value`
  - If `data` is given:
    - Calculate the `lambtha` of `data`
    - If `data` is not a list, raise a TypeError with the message
    `data must be a list`
    - If `data` does not contain at least two data points, raise a `ValueError`
    with the message `data must contain multiple values`
"""


class Poisson:
    """Poisson distribution"""

    def __init__(self, data=None, lambtha=1.0):
        """Class constructor"""
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
