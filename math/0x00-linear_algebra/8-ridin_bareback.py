#!/usr/bin/env python3
"""
Write a function that performs matrix multiplication:
- You can assume that mat1 and mat2 are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be multiplied, return None
"""


def mat_mul(mat1, mat2):
    """Multiply two matrices."""
    if len(mat1[0]) == len(mat2):
        result = [
            [sum(i * j for i, j in zip(row, cols)) for cols in zip(*mat2)]
            for row in mat1
        ]
        return result
    return None
