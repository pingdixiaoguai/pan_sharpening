#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time: 2019/11/20 15:34 
# @Author: PDXG 
# @Site:  
# @File: pca.py 
# @Software: PyCharm
# @description: 该函数用于将HS图片与PAN图片通过pca方法融合

import numpy as np

from interp23tap_general import interp23tap_general
from sklearn.decomposition import PCA


def cs_fusion(hyper_img, pan_img):
    """
    此函数用于将图像进行融合
    :param hyper_img: 高光谱图像在全色图像上的内插（mλ * n）
    :param pan_img: 全色图像 （1 * n）
    :return: 融合后的图像 （mλ * n）
    """

    # 上取样
    ratio1 = pan_img.shape(0) / hyper_img.shape(0)
    hsu = interp23tap_general(hyper_img, ratio1)

    image_lr = hsu
    image_hr = pan_img

    n = image_lr.shape(0)
    m = image_lr.shape(1)
    d = image_lr.shape(2)

    image_lr = np.reshape(image_lr, (n*m, d), 'F')
    PCA()




