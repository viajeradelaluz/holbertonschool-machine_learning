#!/usr/bin/env python3
"""
Write a function that performs addition, subtraction,
multiplication, and division:
- You can assume that mat1 and mat2 can be interpreted as numpy.ndarrays
- Return a tuple containing the sum, difference, product, and quotient
- You are not allowed to use any loops or conditional statements
- You can assume that mat1 and mat2 are never empty
"""


def np_elementwise(mat1, mat2):
    """Perform addition, subtraction, multiplication, and division."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
