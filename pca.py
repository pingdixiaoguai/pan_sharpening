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
    ratio1 = pan_img.shape[0] / hyper_img.shape[0]
    ratio1 = int(ratio1)
    hsu = interp23tap_general(hyper_img, ratio1)

    hyper_image = hsu
    pan_image = pan_img

    n = hyper_image.shape[0]
    m = hyper_image.shape[1]
    d = hyper_image.shape[2]

    # 使用pca方法分解图片，获取weight以及score.
    # 空间信息是集中在第一个主成分上，光谱信息集中在其他主成分上
    # The hypothesis underlying its application to pansharpening
    # is that the spatial information (shared by all the channels) is concentrated in the first PC,
    # while the spectral information (specific to each single band) is accounted for the other PCs
    hyper_image = np.reshape(hyper_image, (n*m, d))
    pca = PCA()
    pca.fit(hyper_image)
    weight = pca.components_.conj().transpose()
    pca_data = pca.fit_transform(hyper_image)

    # 把进行了PCA的hyperspectral图像保存到变量f
    f = np.reshape(pca_data, (n, m, d))

    # 把PAN图标准化，再替代
    i = f[:, :, 1]
    a = pan_image - (np.mean(pan_image.flatten()) * (np.std(i, ddof=1)))
    b = np.std(pan_image.flatten(), ddof=1)
    pan_image = a / b + np.mean(i)

    # Replace 1st band with PAN image
    f[:, :, 1] = pan_image

    # Inverse PCA
    i_fus_pca = np.reshape(f, (n * m, d)) @ (weight.conj().transpose())
    i_fus_pca = np.reshape(i_fus_pca, (n, m, d))

    # 之前这个矩阵是标准化的，现在还原到量纲
    for i in range(hsu.shape[2]):
        h = i_fus_pca[:, :, i]
        i_fus_pca[:, :, i] = h - np.mean(h) + np.mean(np.squeeze(hsu[:, :, i]))

    return i_fus_pca




