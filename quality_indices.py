# encoding: utf-8


"""
@author: PDXG
@file: quality_indices.py
@create_time: 2019/12/24 18:41
@version: 
@description: 
"""
import numpy as np
from pysptools.distance import SAM


def quality_indices(i_hs, i_ref, ratio):
    rows = i_ref.shape[0]
    cols = i_ref.shape[1]

    # 删去边界
    i_hs = i_hs[ratio:rows - ratio, ratio:cols - ratio, :]
    i_ref = i_ref[ratio:rows - ratio, ratio:cols - ratio, :]

    # 重新确定rows, cols, bands
    rows = i_ref.shape[0]
    cols = i_ref.shape[1]
    bands = i_ref.shape[2]

    # 首先计算Cross Correlation(CC)
    # 需要对每个band计算
    out = np.zeros(bands)
    for i in range(bands):
        i_hs_temp = i_hs[:, :, i].flatten()
        i_ref_temp = i_ref[:, :, i].flatten()
        cc = np.corrcoef(i_hs_temp, i_ref_temp)
        out[i] = cc[0, 1]
    cc = np.mean(out)
    print("CC: ", cc)

    # 计算spectral angle mapper(SAM)
    out = np.zeros(bands)
    for i in range(bands):
        i_hs_temp = i_hs[:, :, i].flatten()
        i_ref_temp = i_ref[:, :, i].flatten()
        out[i] = SAM(i_hs_temp, i_ref_temp)
    sam = out.mean()
    print("SAM: ", sam)

    # 计算root mean squared error(RMSE)
    molecule = np.linalg.norm((i_hs - i_ref).reshape(rows * cols, -1), ord='fro')
    denominator = (rows * cols * bands) ** 0.5
    rmse = molecule / denominator
    print("RMSE: ", rmse)

    # 计算erreur relative globale adimensionnelle de synthese(ERGAS)
    err = i_ref - i_hs
    ergas = 0
    for i in range(bands):
        ergas = ergas + np.mean(err[:, :, i] ** 2) / np.mean(i_ref[:, :, i] ** 2)

    ergas = (100 / ratio) * np.sqrt((1 / err.shape[2]) * ergas)
    print("ERGAS: ", ergas)
