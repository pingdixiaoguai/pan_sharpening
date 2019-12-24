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
from scipy.misc import imresize

def cs_fusion(hyper_img, pan_img):
    """
    此函数用于将图像进行融合
    :param hyper_img: 高光谱图像在全色图像上的内插（mλ * n）
    :param pan_img: 全色图像 （1 * n）
    :return: 融合后的图像 （mλ * n）
    """

    ratio1 = pan_img.shape[0] / hyper_img.shape[0]
    ratio1 = int(ratio1)
    hsu = interp23tap_general(hyper_img, ratio1)
    # 上取样

    image_lr = hsu
    image_hr = pan_img

    n = image_lr.shape(0)
    m = image_lr.shape(1)
    d = image_lr.shape(2)

    # 使用pca方法分解图片，获取weight以及score.PCA transform on MS bands
    image_lr = np.reshape(image_lr, (n*m, d), order='F')
    pca = PCA()
    pca.fit(image_lr)
    weight = np.transpose(pca.components_)
    pca_data = pca.fit_transform(image_lr)

    f = np.reshape(pca_data, (n, m, d), order='F')

    # Equalization
    i = f[:, :, 1]
    # matlab a(:) 就是把一个矩阵拉成一个向量
    # matlab a/b : solution of x a = b for x, 在numpy用linalg.lstsq(a,b)
    np.linalg.lstsq(np.std(image_hr.flatten(), ddof=1) + np.mean(i),(image_hr - np.mean(image_hr.flatten())) @ np.std(i, ddof=1))
    # Replace 1st band with PAN.lan
    f[:, :, 1] = image_hr

    # Inverse PCA
    i_fus_pca = np.reshape(f, (n * m, d)) @ (weight.conj().transpose())
    i_fus_pca = np.reshape(i_fus_pca, (n, m, d))

    # Final Linear Equalization
    for i in range(hsu.shape(2)):
        h = i_fus_pca[:, :, i]
        i_fus_pca[:, :, i] = h - np.mean(h) + np.mean(np.squeeze(hsu[:, :, i]))

    return i_fus_pca




