# encoding: utf-8


"""
@author: PDXG
@file: quality_indices.py
@create_time: 2019/12/24 18:41
@version: 
@description: 
"""
import numpy as np
import spectral


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
        molecule = i_hs_temp.T @ i_ref_temp
        denominator = np.linalg.norm(i_hs_temp) * np.linalg.norm(i_ref_temp)
        out[i] = np.arccos(molecule / denominator)
    sam = np.mean(out)
    print("SAM: ", sam)

    # 计算root mean squared error(RMSE)
    temp = i_hs - i_ref
    out = np.zeros(bands)
    for i in range(bands):
        molecule = np.linalg.norm(temp[:, :, i], ord='fro')
        denominator = (rows * cols * bands) ** 0.5
        out[i] = molecule / denominator
    rmse = np.mean(out)
    print("RMSE: ", rmse)

    # 计算erreur relative globale adimensionnelle de synthese(ERGAS)
    err = i_ref - i_hs
    ergas = 0
    for i in range(bands):
        ergas = ergas + np.linalg.norm(err[:, :, i], ord='fro') / np.linalg.norm(i_hs[:, :, i], ord='fro')

    ergas = (100 / ratio) * np.sqrt((1 / err.shape[2]) * ergas)
    print("ERGAS: ", ergas)
