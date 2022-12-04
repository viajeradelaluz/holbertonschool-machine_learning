#!/usr/bin/env python3

def matrix_shape(matrix):
    """Calculates the shape of a matrix.
    - You can assume elements in the same dimension are of the same type/shape
    - The shape should be returned as a list of integers
    """
    shape = []
    while type(matrix) == list and len(matrix) != 0:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
