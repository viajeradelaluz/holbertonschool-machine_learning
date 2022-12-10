#!/usr/bin/env python3
"""
Write a function that adds two arrays element-wise:
- You can assume that arr1 and arr2 are lists of ints/floats
- You must return a new list
- If arr1 and arr2 are not the same shape, return None
"""


def add_arrays(arr1, arr2):
    """Add two arrays of the same shape."""
    if len(arr1) != len(arr2):
        return None
    result = map(lambda x, y: x + y, arr1, arr2)
    return list(result)
