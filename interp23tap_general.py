#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time: 2019/11/22 15:09 
# @Author: PDXG 
# @Site:  
# @File: interp23tap_general.py 
# @Software: PyCharm
# @description:
import numpy as np

from scipy.misc import imresize


def interp23tap_general(i_interpolated, ratio):
    """
    做双三次插值
    """
    r = i_interpolated.shape[0]
    c = i_interpolated.shape[1]
    b = i_interpolated.shape[2]
    i1_lru = np.zeros((ratio * r, ratio * c, b))
    for i in range(b):
        i1_lru[:, :, i] = imresize(i_interpolated[:, :, i], size=(ratio * r, ratio * c),
                                   interp='bicubic')
    i_interpolated = i1_lru
    return i_interpolated
