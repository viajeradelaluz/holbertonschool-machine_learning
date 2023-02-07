#!/usr/bin/env python3
"""
Module that performs back propagation over a pooling layer of a neural network.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """Performs back propagation over a pooling layer of a neural network.

    :param dA: a np array (m, h_new, w_new, c_new) with the partial derivatives
    with respect to the output of the pooling layer.
        m: the number of examples.
        h_new: the height of the output.
        w_new: the width of the output.
        c_new: the number of channels in the output.
    :param A_prev: a np array (m, h_prev, w_prev, c_prev) with the output of
    the previous layer.
        h_prev: the height of the previous layer.
        w_prev: the width of the previous layer.
    :param kernel_shape: a tuple of (kh, kw) with the size of the kernel.
        kh: the kernel height.
        kw: the kernel width.
    :param stride: a tuple of (sh, sw) with the strides for the convolution.
        sh: the stride for the height.
        sw: the stride for the width.
    :param mode: a string with the type of pooling, either max or avg.

    :return: (dA_prev) partial derivatives with respect to the previous layer.
    """

    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for init in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    xa, ya = (x * sh, y * sw)
                    xb, yb = (xa + kh, ya + kw)

                    if mode == "max":
                        A = A_prev[init, xa:xb, ya:yb, z]
                        layer = (A == np.max(A)) * dA[init, x, y, z]
                        dA_prev[init, xa:xb, ya:yb, z] += layer

                    if mode == "avg":
                        dA_avg = dA[init, x, y, z] / kh / kw
                        layer = np.ones((kh, kw)) * dA_avg
                        dA_prev[init, xa:xb, ya:yb, z] += layer

    return dA_prev
