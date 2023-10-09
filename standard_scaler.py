#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/10/09 10:48

@Author: Sun Jiahua

@File  : standard_scaler.py

@Desc  : 在训练模型时，对数据进行z-score处理时，要将相关参数保存下来，以便在线预测时调用
"""
import numpy as np


class StandardScaler():

    def __init__(self, mean=None, std=None):
        self.mean_ = mean
        self.std_ = std

    def fit(self, x):
        x = np.array(x)
        assert x.ndim == 2, 'x的维度应该为(m, n), 其中m为样本数，n为特征数'

        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)

        return self

    def transform(self, x):
        x = np.array(x)
        assert x.ndim == 2, 'x的维度应该为(m, n), 其中m为样本数，n为特征数'

        x = x.astype(np.float)
        x -= self.mean_
        x /= self.std_

        return x

    def inverse_transform(self, x_scaled):
        x_scaled = np.array(x_scaled)
        assert x_scaled.ndim == 2, 'x的维度应该为(m, n), 其中m为样本数，n为特征数'

        x_scaled = x_scaled.astype(np.float)
        x_scaled *= self.std_
        x_scaled += self.mean_
        return x_scaled
