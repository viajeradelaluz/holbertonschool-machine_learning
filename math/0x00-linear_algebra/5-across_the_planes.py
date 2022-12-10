#!/usr/bin/env python3
"""
Write a function that adds two matrices element-wise:
- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If mat1 and mat2 are not the same shape, return None
"""


def add_matrices2D(mat1, mat2):
    """Add two matrices of the same shape."""
    if len(mat1[0]) != len(mat2[0]):
        return None

    result, i = [], 0
    while i < len(mat1):
        addition = list(map(lambda x, y: x + y, mat1[i], mat2[i]))
        result.append(addition)
        i += 1

    return result
