#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/2 18:43

@Author: Sun Jiahua

@File  : feature_extraction.py

@Desc  : 特征提取，提取16个特征
"""

import numpy as np


def feature_extraction(data):
    li = []
    size = len(data)

    # 最大值
    max1 = np.max(data)
    li.append(max1)

    # 平均值
    mean1 = np.mean(data)
    li.append(mean1)

    # 最小值
    min1 = np.min(data)
    li.append(min1)

    # 标准差
    std_value = np.std(data)
    li.append(std_value)

    # 峰峰值
    P_Pvalue = max1 - min1
    li.append(P_Pvalue)

    # 平均幅值
    Xmean = np.mean(np.abs(data))
    li.append(Xmean)

    # 均方根值
    root_mean_score = np.sqrt(np.sum(np.square(data)) / size)
    li.append(root_mean_score)

    # 歪度值
    skewness = np.sum(np.power(data, 3)) / size
    li.append(skewness)
    # 峭度值
    Kurtosis_value = np.sum(np.power(data, 4)) / size
    li.append(Kurtosis_value)

    # 波形指标
    absolute_mean_value = np.sum(np.fabs(data)) / size  # 绝对平均值
    shape_factor = root_mean_score / absolute_mean_value
    li.append(shape_factor)

    # 脉冲指标
    pulse_factor = max1 / absolute_mean_value
    li.append(pulse_factor)

    # 歪度指标
    Kurtosis_factor = Kurtosis_value / root_mean_score
    li.append(Kurtosis_factor)

    # 峰值指标
    crest_factor = max1 / root_mean_score
    li.append(crest_factor)

    # 裕度指标
    Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data))) / size)  # 方根幅值
    clearance_factor = max1 / Root_amplitude
    li.append(clearance_factor)

    # 峭度指标
    Kurtosis_factor = Kurtosis_value / np.power(root_mean_score, 4)
    li.append(Kurtosis_factor)

    # 方根幅值
    Xr = np.square(np.mean(np.sqrt(np.abs(data))))
    li.append(Xr)

    return li