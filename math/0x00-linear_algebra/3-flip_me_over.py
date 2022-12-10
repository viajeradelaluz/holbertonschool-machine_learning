#!/usr/bin/env python3
"""
Write a function that returns the transpose of a 2D matrix, matrix:
- You must return a new matrix
- You can assume that matrix is never empty
- You can assume all elements in the same dimension are of the same type/shape
"""


def matrix_transpose(matrix):
    matrix_t, j = [], 0
    while j < len(matrix[0]):
        row, i = [], 0
        while i < len(matrix):
            row.append(matrix[i][j])
            i += 1
        matrix_t.append(row)
        j += 1
    return matrix_t
