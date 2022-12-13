#!/usr/bin/env python3
"""
Write a function that calculates the derivative of a polynomial:
- poly is a list of coefficients representing a polynomial
  - the index of the list is the power of x that the coefficient belongs to
  - Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
- If poly is not valid, return None
- If the derivative is 0, return [0]
- Return a new list of coefficients as the derivative of the polynomial
"""


def poly_derivative(poly):
    """Returns the derivative of a polynomial"""
    if isinstance(poly, list) and len(poly) > 0:
        if len(poly) == 1:
            return [0]
        coefficients = [poly[i] * i for i in range(1, len(poly))]
        return coefficients
    return None
