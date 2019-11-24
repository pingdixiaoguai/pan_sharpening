#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time: 2019/11/22 15:09 
# @Author: PDXG 
# @Site:  
# @File: interp23tap_general.py 
# @Software: PyCharm
# @description:
import numpy as np
import scipy.ndimage

from scipy import signal


def interp23tap_general(i_interpolated, ratio):
    tap = 44
    r = i_interpolated.shape(0)
    c = i_interpolated.shape(1)
    b = i_interpolated.shape(2)

    base_cut_off = ratio * signal.firwin(tap, 1 / ratio)

    # 创建一个ratio*r × ratio*c × b大小的矩阵
    i1_lru = np.zeros((ratio * r, ratio * c, b))

    for i in range(b):
        t = i1_lru[:, :, i]
        # matlab中的imfilter对2维矩阵等价于scipy.ndimage.correlate，参数circular等价于wrap
        t = scipy.ndimage.correlate(t.T, base_cut_off, mode='wrap').conj().transpose()
        i1_lru[:, :, i] = scipy.ndimage.correlate(t.T, base_cut_off, mode='wrap').conj().transpose()

    i_interpolated = i1_lru
    return i_interpolated
