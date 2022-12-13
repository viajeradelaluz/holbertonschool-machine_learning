#!/usr/bin/env python3
"""
Write a function that calculates the sumof i squared:
- Where i starts at 1 and ends at n.
"""


def summation_i_squared(n):
    """Returns summation of i squared"""
    if isinstance(n, int) and n > 0:
        return sum(map(lambda n: n**2, range(n + 1)))
    return None
