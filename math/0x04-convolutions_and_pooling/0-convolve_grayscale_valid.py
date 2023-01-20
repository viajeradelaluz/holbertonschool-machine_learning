#!/usr/bin/env python3
"""
Module that performs a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    :param images: np array shape (m, h, w), with multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    :param kernel: np array shape (kh, kw), with kernel for the convolution
        - kh: height
        - kw: width

    :return: a np array with the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    h_output, w_output = h - kh + 1, w - kw + 1
    convolved_image = np.zeros((m, h_output, w_output))

    for x in range(h_output):
        for y in range(w_output):
            x0 = x + kh
            y0 = y + kw
            convolved_image[:, x, y] = np.sum(
                images[:, x:x0, y:y0] * kernel, axis=(1, 2)
            )

    return convolved_image
