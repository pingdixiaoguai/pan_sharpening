# encoding: utf-8


"""
@author: PDXG
@file: quality_indices.py
@create_time: 2019/12/24 18:41
@version: 
@description: 
"""
import numpy as np


def quality_indices(i_hs, i_ref, ratio):
    rows = i_ref.shape[0]
    cols = i_ref.shape[1]
    bands = i_ref.shape[2]

    # 首先计算Cross Correlation
    out = np.zeros(bands)
    for i in range(bands):
        i_hs_temp = i_hs[ratio:rows - ratio, ratio:cols - ratio, i].flatten()
        i_ref_temp = i_ref[ratio:rows - ratio, ratio:cols - ratio, i].flatten()
        cc = np.corrcoef(i_hs_temp, i_ref_temp)
        out = cc[0, 1]
    out = np.mean(out)
    print("CC: ", out)
