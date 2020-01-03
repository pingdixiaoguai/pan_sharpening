#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time: 2020/1/2 14:56 
# @Author: PDXG 
# @Site:  
# @File: conv_downsample.py 
# @Software: PyCharm
# @description:

import numpy as np
from scipy.ndimage import correlate


def matlab_style_gauss2d(shape, sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def conv_downsample(i_ref, ratio, size_kernel, sig):
    blu_ker = matlab_style_gauss2d(size_kernel, sig)

    down_sample_image = np.zeros(i_ref.shape)
    for i in range(i_ref.shape[2]):
        down_sample_image[:, :, i] = correlate(i_ref[:, :, i], blu_ker, mode='wrap')
    down_sample_image = down_sample_image[::ratio, ::ratio, :]
    return down_sample_image

