#!/usr/bin/env python3
"""
Write a function that calculates the shape of a matrix:
  - You can assume elements in the same dimension are of the same type/shape
  - The shape should be returned as a list of integers
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix."""
    shape = []
    while type(matrix) == list and len(matrix) != 0:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
