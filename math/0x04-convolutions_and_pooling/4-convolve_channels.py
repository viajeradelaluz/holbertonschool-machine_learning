#!/usr/bin/env python3
"""
Module that performs a convolution on images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding="same", stride=(1, 1)):
    """Performs a convolution on images with channels.

    :param images: np array shape (m, h, w, c), with multiple grayscale images
        - m: number of images
        - h: height in pixels
        - w: width in pixels
        - c: number of channels
    :param kernel: np array shape (kh, kw, c), with kernel for the convolution
        - kh: height
        - kw: width
    :param padding: tuple of (ph, pw)
        - if "same", performs a same convolution
        - if "valid", performs a valid convolution
        - if a tuple:
            - ph: padding for the height
            - pw: padding for the width
        - the image should be padded with 0's
    :param stride: tuple of (sh, sw)
        - sh: stride for the height of image
        - sw: stride for the width of image

    :return: a np array with the convolved images
    """
    m, h, w, _ = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Pad dimensions
    if padding == "same":
        h_pad = ((h - 1) * sh + kh - h) // 2 + 1
        w_pad = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == "valid":
        h_pad = 0
        w_pad = 0
    else:
        h_pad, w_pad = padding

    image_pad = np.pad(
        images,
        ((0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0)),  # type: ignore
        "constant",  # type: ignore
    )

    output_h = (h + 2 * h_pad - kh) // sh + 1
    output_w = (w + 2 * w_pad - kw) // sw + 1
    convolved_image = np.zeros((m, output_h, output_w))

    for x in range(output_h):
        for y in range(output_w):
            x0 = x * sh
            x1 = x0 + kh
            y0 = y * sw
            y1 = y0 + kw
            convolved_image[:, x, y] = np.sum(
                image_pad[:, x0:x1, y0:y1, :] * kernel, axis=(1, 2, 3)
            )

    return convolved_image