#!/usr/bin/env python3
"""
Module that performs a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

    :param images: np array shape (m, h, w), with multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
    :param kernel: np array shape (kh, kw), with kernel for the convolution
        - kh: height
        - kw: width
    :param padding: tuple of (ph, pw)
        - ph: padding for the height
        - pw: padding for the width
        - the image should be padded with 0's

    :return: a np array with convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    h_pad, w_pad = padding
    image_pad = np.pad(
        images,
        ((0, 0), (h_pad, h_pad), (w_pad, w_pad)),
        "constant",
    )

    h_output, w_output = h + 2 * h_pad - kh + 1, w + 2 * w_pad - kw + 1
    convolved_image = np.zeros((m, h_output, w_output))

    for x in range(h_output):
        for y in range(w_output):
            x0 = x + kh
            y0 = y + kw
            convolved_image[:, x, y] = np.sum(
                image_pad[:, x:x0, y:y0] * kernel, axis=(1, 2)
            )

    return convolved_image
