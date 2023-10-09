#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/3 14:06

@Author: Sun Jiahua

@File  : diagnosis.py

@Desc  : 故障诊断的相关函数
"""
import numpy as np
import joblib
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
import keras.backend as K

from feature_extraction import feature_extraction


def diagnosis(diagnosis_samples, model_file_path):
    """
    故障诊断
    :param diagnosis_samples: 数据样本
    :param model_file_path: 模型路径
    :return: pred_result：诊断结果
    """
    suffix = model_file_path.split('/')[-1].split('.')[-1]  # 获得所选模型的后缀名
    if 'm' == suffix:  # 说明是随机森林
        # 提取特征
        loader = np.empty(shape=[diagnosis_samples.shape[0], 16])
        for i in range(diagnosis_samples.shape[0]):
            loader[i] = feature_extraction(diagnosis_samples[i])
        diagnosis_samples_feature_extraction = loader

        # 加载模型
        model = joblib.load(model_file_path)
        # 使用模型进行诊断
        y_preds = model.predict(diagnosis_samples_feature_extraction)
    else:
        K.clear_session()
        diagnosis_samples_new = diagnosis_samples[:, :, np.newaxis]  # 添加一个新维度
        # 加载模型 --- 这里要用这种方法加载，不然加载有的模型会报错，我也不知道为什么
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model(model_file_path)
        # 对于CNN模型和LSTM,GRU模型，两者的输入不相同，所以捕捉一下异常，如果上面那种维度错了，那就换一个维度
        try:
            y_preds = model.predict_classes(diagnosis_samples_new)
        except ValueError:
            diagnosis_samples_new = diagnosis_samples[:, np.newaxis, :]  # 添加一个新维度
            y_preds = model.predict_classes(diagnosis_samples_new)

    y_preds = list(y_preds)
    # 计算这些样本诊断结果中出现次数最多的结果作为最后结果
    y_pred = max(y_preds, key=y_preds.count)
    pred_result = result_decode(y_pred)

    return pred_result


def result_decode(y_pred):
    '''
    将数字表示的诊断结果解码为文字
    :param y_pred:
    :return:
    '''
    if 0 == y_pred:
        pred_result = '滚动体故障：0.1778mm'
    elif 1 == y_pred:
        pred_result = '滚动体故障：0.3556mm'
    elif 2 == y_pred:
        pred_result = '滚动体故障：0.5334mm'
    elif 3 == y_pred:
        pred_result = '内圈故障：0.1778mm'
    elif 4 == y_pred:
        pred_result = '内圈故障：0.3556mm'
    elif 5 == y_pred:
        pred_result = '内圈故障：0.5334mm'
    elif 6 == y_pred:
        pred_result = '外圈故障（6点方向）：0.1778mm'
    elif 7 == y_pred:
        pred_result = '外圈故障（6点方向）：0.3556mm'
    elif 8 == y_pred:
        pred_result = '外圈故障（6点方向）：0.5334mm'
    elif 9 == y_pred:
        pred_result = '正常'

    return pred_result