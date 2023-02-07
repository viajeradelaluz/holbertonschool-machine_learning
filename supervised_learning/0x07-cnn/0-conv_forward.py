#!/usr/bin/env python3
"""
Module that performs forward propagation over a convolutional layer of a neural
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural
    network.

    :param A_prev: a np array (m, h_prev, w_prev, c_prev) with the output of
    the previous layer.
        m: the number of examples.
        h_prev: the height of the previous layer.
        w_prev: the width of the previous layer.
        c_prev: the number of channels in the previous layer.
    :param W: a np array (kh, kw, c_prev, c_new) with the kernels for the
    convolution.
        kh: the filter height.
        kw: the filter width.
        c_prev: the number of channels in the previous layer.
        c_new: the number of channels in the output.
    :param b: a np array (1, 1, 1, c_new) with the biases applied to the
    convolution.
    :param activation: an activation function applied to the convolution.
    :param padding: a string that is either same or valid, indicating the type
    of padding used.
    :param stride: a tuple of (sh, sw) with the strides for the convolution.
        sh: the stride for the height.
        sw: the stride for the width.

    :return: the output of the convolutional layer.
    """

    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    pad_h, pad_w = 0, 0
    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2

    pad = np.pad(
        array=A_prev,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )

    conv_h = (h_prev + 2 * pad_h - kh) // sh + 1
    conv_w = (w_prev + 2 * pad_w - kw) // sw + 1
    convolution = np.zeros((m, conv_h, conv_w, c_new))

    for x in range(conv_h):
        for y in range(conv_w):
            for z in range(c_new):
                xa, ya = (x * sh, y * sw)
                xb, yb = (xa + kh, ya + kw)
                A = pad[:, xa:xb, ya:yb, :]

                convolution[:, x, y, z] = np.sum(
                    A * W[..., z],
                    axis=(1, 2, 3),
                )

    return activation(convolution + b)
