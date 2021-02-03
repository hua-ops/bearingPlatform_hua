#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/2 13:08

@Author: Sun Jiahua

@File  : preprocess_train_result.py

@Desc  : 对模型的训练结果进行处理：
            训练集和验证集损失及正确率曲线
            绘制混淆矩阵
            分类报告
            绘制 ROC曲线，精度召回曲线
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import scikitplot as skplt


def plot_history_curcvs(history, save_path, model_name):
    '''
    绘制 训练集 和 验证集 的 损失 及 正确率 曲线
    :param history: 模型训练（fit)的返回参数
    :param save_path: 生成图片的保存路径
    :param model_name: 模型名称
    :return:
    '''
    acc = history.history['acc']  # 每一轮 在 训练集 上的 精度
    val_acc = history.history['val_acc']  # 每一轮 在 验证集 上的 精度
    loss = history.history['loss']  # 每一轮 在 训练集 上的 损失
    val_loss = history.history['val_loss']  # 每一轮 在 验证集 上的 损失

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_acc.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()  # 再画一个图，显式 创建figure对象
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()


def plot_confusion_matrix(model, model_name, save_path, X_test, y_test):
    '''
    绘制混淆矩阵
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    if '1D_CNN' == model_name:
        X_test = X_test[:, :, np.newaxis]  # 添加一个新的维度
    elif 'LSTM' == model_name or 'GRU' == model_name:
        X_test = X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 随机森林不需要添加维度

    # 这里两种的 预测函数 不同
    if 'random_forest' == model_name:
        y_preds = model.predict(X_test)
    else:
        y_preds = model.predict_classes(X_test)

    y_test = [np.argmax(item) for item in y_test]  # one-hot解码

    # 绘制混淆矩阵
    con_mat = confusion_matrix(y_test, y_preds)

    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)  # np.around(): 四舍五入

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm,
                annot=True,  # annot: 默认为False，为True的话，会在格子上显示数字
                cmap='Blues'  # 热力图颜色
                )

    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(save_path + '/' + model_name + '_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()


def brief_classification_report(model, model_name, X_test, y_test):
    '''
    计算 分类报告
    :param model: 模型
    :param model_name:  模型名称
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return: classification_report：分类报告
    '''
    if '1D_CNN' == model_name:
        X_test = X_test[:, :, np.newaxis]  # 添加一个新的维度
    elif 'LSTM' == model_name or 'GRU' == model_name:
        X_test = X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 随机森林不需要添加维度

    # 这里两种的 预测函数 不同
    if 'random_forest' == model_name:
        y_preds = model.predict(X_test)
    else:
        y_preds = model.predict_classes(X_test)

    y_test = [np.argmax(item) for item in y_test]  # one-hot解码
    classification_report = metrics.classification_report(y_test, y_preds)

    return classification_report


def plot_metrics(model, model_name, save_path, X_test, y_test):
    '''
    绘制 ROC曲线 和 精度召回曲线
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    if '1D_CNN' == model_name:
        X_test = X_test[:, :, np.newaxis]  # 添加一个新的维度
    elif 'LSTM' == model_name or 'GRU' == model_name:
        X_test = X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 随机森林不需要添加维度

    y_probas = model.predict_proba(X_test)
    y_test = [np.argmax(item) for item in y_test]  # one-hot解码

    # 绘制“ROC曲线”
    skplt.metrics.plot_roc(y_test, y_probas, title=model_name+' ROC Curves', figsize=(7, 7),
                           # title_fontsize = 24, text_fontsize = 16
                           )
    plt.savefig(save_path + '/' + model_name + '_ROC_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制“精度召回曲线”
    skplt.metrics.plot_precision_recall(y_test, y_probas, title=model_name+' Precision-Recall Curves', figsize=(7, 7),
                                        # title_fontsize = 24, text_fontsize = 16
                                        )
    plt.savefig(save_path + '/' + model_name + '_Precision_Recall_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
