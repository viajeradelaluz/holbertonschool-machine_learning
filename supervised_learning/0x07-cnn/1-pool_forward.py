#!/usr/bin/env python3
"""
Module that performs forward propagation over a pooling layer of a neural
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """Performs forward propagation over a pooling layer of a neural network.

    :param A_prev: a np array (m, h_prev, w_prev, c_prev) with the output of
    the previous layer.
        m: the number of examples.
        h_prev: the height of the previous layer.
        w_prev: the width of the previous layer.
        c_prev: the number of channels in the previous layer.
    :param kernel_shape: a tuple of (kh, kw) with the size of the kernel
        kh: the kernel height.
        kw: the kernel width.
    :param stride: a tuple of (sh, sw) with the strides for the convolution.
        sh: the stride for the height.
        sw: the stride for the width.
    :param mode: a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively.

    :return: the output of the pooling layer.
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pool_h = (h_prev - kw) // sh + 1
    pool_w = (w_prev - kw) // sw + 1
    pool_mode = np.max if mode == "max" else np.average
    pooling = np.zeros((m, pool_h, pool_w, c_prev))

    for x in range(pool_h):
        for y in range(pool_w):
            xa, ya = (x * sh, y * sw)
            xb, yb = (xa + kh, ya + kw)

            pooling[:, x, y, :] = pool_mode(
                A_prev[:, xa:xb, ya:yb, :],
                axis=(1, 2),
            )

    return pooling
