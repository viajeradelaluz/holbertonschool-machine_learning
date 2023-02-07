#!/usr/bin/env python3
"""
Module that performs back propagation over a convolutional layer of a neural.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a neural network

    :param dZ: a np array (m, h_new, w_new, c_new) with the partial derivatives
    with respect to the unactivated output of the convolutional layer.
        m: the number of examples.
        h_new: the height of the output.
        w_new: the width of the output.
        c_new: the number of channels in the output.
    :param A_prev: a np array (m, h_prev, w_prev, c_prev) with the output of
    the previous layer.
        h_prev: the height of the previous layer.
        w_prev: the width of the previous layer.
        c_prev: the number of channels in the previous layer.
    :param W: a np array (kh, kw, c_prev, c_new) with the kernels
        kh: the filter height.
        kw: the filter width.
    :param b: a np array (1, 1, 1, c_new) with the biases
    :param padding: a string that is either same or valid, indicating the type
    of padding used.
    :param stride: a tuple of (sh, sw) with the strides for the convolution.
        sh: the stride for the height.
        sw: the stride for the width.

    :return: a tuple of (dA_prev, dW, db) containing the partial derivatives,
    the kernels, and the biases, respectively.

    """

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    pad_h, pad_w = 0, 0
    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_pad = np.pad(
        array=A_prev,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )
    dA_pad = np.pad(
        array=dA_prev,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )

    for init in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    xa, ya = (x * sh, y * sw)
                    xb, yb = (xa + kh, ya + kw)
                    A = dZ[init, x, y, z]

                    dA_pad[init, xa:xb, ya:yb, :] += A * W[..., z]
                    dW[..., z] += A * A_pad[init, xa:xb, ya:yb, :]

    h, w = (slice(pad_h, -pad_h), slice(pad_w, -pad_w))
    dA_prev = dA_pad[:, h, w, :] if padding == "same" else dA_pad

    return dA_prev, dW, db
