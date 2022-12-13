#!/usr/bin/env python3
"""
Write a function that calculates the integral of a polynomial:
- poly is a list of coefficients representing a polynomial
  - the index of the list is the power of x that the coefficient belongs to
  - Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
- C is an integer representing the integration constant
- If a coefficient is a whole number, it should be represented as an integer
- If poly or C are not valid, return None
- Return a new list of coefficients of the integral of the polynomial
- The returned list should be as small as possible
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if isinstance(poly, list) and isinstance(C, int) and len(poly) > 0:
        if len(poly) == 1 and poly[0] == 0:
            return [C]
        coefficients = [C]
        for i in range(len(poly)):
            coefficients.append(poly[i] / (i + 1))
        return coefficients
    return None
