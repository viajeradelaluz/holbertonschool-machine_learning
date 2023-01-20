#!/usr/bin/env python3
"""
Module that performs pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode="max"):
    """Performs pooling on images.

    :param images: np array shape (m, h, w, c), with multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
        - c: number of channels
    :param kernel_shape: np array (kh, kw), with kernel for the convolution
        - kh: height
        - kw: width
    :param stride: tuple of (sh, sw)
        - sh: stride for the height of image
        - sw: stride for the width of image
    :param mode: indicates the type of pooling
        - max: max pooling
        - avg: average pooling

    :return: a np array with the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_output = (h - kw) // sh + 1
    w_output = (w - kw) // sw + 1
    pooled_image = np.zeros((m, h_output, w_output, c))

    if mode == "max":
        pool_function = np.max
    else:
        pool_function = np.average

    for x in range(h_output):
        for y in range(w_output):
            x0 = x * sh
            x1 = x0 + kh
            y0 = y * sw
            y1 = y0 + kw
            pooled_image[:, x, y, :] = pool_function(
                images[:, x0:x1, y0:y1, :], axis=(1, 2)
            )

    return pooled_image
