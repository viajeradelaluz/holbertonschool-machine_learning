#!/usr/bin/env python3
"""
Module that performs a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    :param images: np array shape (m, h, w), with multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    :param kernel: np array shape (kh, kw), with kernel for the convolution
        - kh: height
        - kw: width

    :return: a np array with the convolved images
    """
    _, h, w = images.shape
    kh, kw = kernel.shape

    h_pad, w_pad = kh // 2, kw // 2
    image_pad = np.pad(
        images,
        ((0, 0), (h_pad, h_pad), (w_pad, w_pad)),
        "constant",
    )

    convolved_image = np.zeros(images.shape)

    for x in range(h):
        for y in range(w):
            x0 = x + kh
            y0 = y + kw
            convolved_image[:, x, y] = np.sum(
                image_pad[:, x:x0, y:y0] * kernel, axis=(1, 2)
            )

    return convolved_image
