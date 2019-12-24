#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time: 2019/12/16 15:05 
# @Author: PDXG 
# @Site:  
# @File: conv_downsample.py 
# @Software: PyCharm
# @description: 
import scipy.ndimage
import numpy as np


def matlab_style_gauss2D(shape,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(- (x * x + y * y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def conv_downsample(ref_image, ratio, size_kernel, sig):
    """
    Input：
        ref_image: 引用图像
        ratio: 下采样因子
        size_kernel: size_kernel[0]和size_kernel[1]分表代表模糊核的高与长

    output：
        blurred_image: 模糊图
        bluker: 模糊核
    """
    # 求模糊核
    bluker = matlab_style_gauss2D(size_kernel, sig)
    blurred_image = np.zeros(ref_image.shape)
    # 因为python只能对每个band求,这里只能用循环实现滤波
    for i in range(ref_image.shape[2]):
        blurred_image_each_band = scipy.ndimage.convolve(ref_image[:, :, i], bluker, mode='wrap')
        blurred_image[:, :, i] = blurred_image_each_band

    # 每5行5列取一个
    blurred_image = blurred_image[::ratio, ::ratio, :]

    return blurred_image, bluker
