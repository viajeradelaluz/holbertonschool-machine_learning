#!/usr/bin/env python3

mat_mul = __import__("8-ridin_bareback").mat_mul

mat1 = [[1, 2], [3, 4], [5, 6]]
mat2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
print(mat_mul(mat1, mat2))

a = [[1, 3, 6], [4, 7, -1], [5, 3, 2]]
b = [[-1, 3, -5, -6], [2, 1, -7, 2], [6, 4, 9, 1]]
print(mat_mul(a, b))
