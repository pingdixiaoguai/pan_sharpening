#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time: 2019/11/22 14:55 
# @Author: PDXG 
# @Site:  
# @File: main.py 
# @Software: PyCharm
# @description:

import struct
import numpy as np

from pca import cs_fusion
from conv_downsample import conv_downsample

def load_binary_file(filename):
    """
    用于读取二进制文件的函数
    """
    # 二进制文件，读取使用rb
    with open(filename, mode='rb') as file:
        temp = file.read()
    file.close()

    # 使用unpack函数将二进制数据转码，注意使用的读取是小端存储的，而且是double类型8字节转码
    data = struct.unpack("<" + "d" * (len(temp) // 8), temp[:])

    return data


if __name__ == '__main__':
    # 读取REF文件
    i_ref = load_binary_file('REF')
    i_ref = np.asarray(i_ref)
    # 数据是395*185*176的，因此读出来应该是一个12861200大小的1维向量
    # 将数据转换成395*185*176的以便处理,matlab转换方式是Fortran,所以order='F'
    i_ref = np.reshape(i_ref, (395, 185, 176), order='F')

    ratio = 5
    overlap = np.asarray([x for x in range(1, 42)])
    size_kernel = (9, 9)
    sig = (1 / (2 * 2.7725887 / ratio ** 2)) ** 0.5

    i_hs, ker_blu = conv_downsample(i_ref, ratio, size_kernel, sig)
    print(i_hs.shape)
    print(i_hs[:, :, 0])
