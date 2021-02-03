#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/2 11:30

@Author: Sun Jiahua

@File  : data_preprocess.py

@Desc  : 数据处理的相关函数
"""

from scipy.io import loadmat
import numpy as np
import os

from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


def training_stage_prepro(data_path, signal_length=864, signal_number=1000, normal=True, rate=[0.7, 0.2, 0.1], enhance=True,
                          enhance_step=28):
    """
    函数说明：训练阶段对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本

    Parameters：
        data_path : string，数据集路径
        signal_length : int, 每次处理的信号长度，默认2个信号周期，864
        signal_number : int, 每个文件（类别）要抽取的信号个数。默认每个类别抽取1000个数据
        normal : bool, 是否标准化。默认True
        rate : list, 训练集/验证集/测试集比例. 默认[0.5,0.25,0.25]
        enhance : bool, 训练集是否采用数据增强. 默认True
        enhance_step : int, 增强数据集采样顺延间隔
    Returns:
        X_train : 训练集
        y_train : 训练集标签
        X_valid : 验证集
        y_valid : 验证集标签
        X_test : 测试集
        y_test : 测试集标签
    """
    # 获得该文件夹下所有.mat文件名
    file_names = os.listdir(data_path)

    def capture():
        """
        函数说明：读取mat文件，并将数据以字典返回（文件名作为key，数据作为value）

        Parameters:
            无
        Returns:
            data_dict : 数据字典
        """
        data_dict = {}

        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)  # 文件路径
            file = loadmat(file_path)  # 读取 .mat 文件，返回的是一个 字典
            file_keys = file.keys()  # 获得该字典所有的key
            for key in file_keys:  # 遍历key, 获得 DE 的数据
                if 'DE' in key:  # DE: 驱动端加速度数据
                    data_dict[file_name] = file[key].ravel()
        return data_dict

    def slice_enhance(data_dict, slice_rate=rate[1] + rate[2]):
        """
        函数说明：将数据分为 训练集 和 <验证及测试集>，并对 训练集 数据进行增强

        Parameters：
            data_dict : dict, 要进行划分的数据
            slice_rate: <验证集以及测试集>所占的比例
        Returns:
            train_samples : 切分后的 训练样本
            valid_test_samples : 切分后的 <验证及测试>样本
        """
        train_samples = {}  # 训练集 样本
        valid_test_samples = {}  # 验证及测试集 样本

        keys = data_dict.keys()
        for key in keys:  # 遍历 数据字典，即每一个文件
            slice_data = data_dict[key]  # 获得value，即取得该文件里的 DE 数据
            all_lenght = len(slice_data)  # 获得数据长度
            end_index = int(all_lenght * (1 - slice_rate))  # 得到 训练集 结束的位置（索引）
            train_samples_num = int(signal_number * (1 - slice_rate))  # 训练集信号 个数
            train_sample = []  # 该文件中的 训练集 样本
            valid_test_sample = []  # 该文件中的 验证及测试集 样本
            if enhance:  # 使用数据增强
                enc_time = signal_length // enhance_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(train_samples_num):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * signal_length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enhance_step
                        sample = slice_data[random_start: random_start + signal_length]
                        train_sample.append(sample)
                        if samp_step == train_samples_num:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(train_samples_num):  # 在该文件中 抽取 训练集 的 信号，共抽取train_samples_num个（随机抽取）
                    random_start = np.random.randint(low=0, high=(
                                end_index - signal_length))  # high=(end_index - signal_length)：保证从任何一个位置开始都可以取到完整的数据长度
                    sample = slice_data[random_start: random_start + signal_length]  # 抽取信号
                    train_sample.append(sample)  # 将抽取到的信号加入 train_samples

            # 抓取测试数据
            for h in range(signal_number - train_samples_num):  # signal_number - train_samples_num：验证和测试集信号个数
                random_start = np.random.randint(low=end_index, high=(all_lenght - signal_length))
                sample = slice_data[random_start: random_start + signal_length]
                valid_test_sample.append(sample)
            train_samples[key] = train_sample  # 字典存储---文件名：对应的信号
            valid_test_samples[key] = valid_test_sample
        return train_samples, valid_test_samples

    def add_labels(data_set_dict):
        '''
        函数说明：为抽样完成的数据添加标签。（这里将字典里每个key的value都append到了一个list里，也就是变成了 一维 数据)

        Parameters:
            data_set_dict : dict, 抽样完成的数据，类型为字典（文件名作为key，数据作为value）
        Returns:
            X : 数据
            Y : 标签

        '''
        X = []  # 数据
        Y = []  # 标签
        label = 0
        for file_name in file_names:
            x = data_set_dict[file_name]  # 取得对应文件中样本的数据
            X += x  # 将其放到 数据表 X 中。这里的 += 等价于 append
            lenx = len(x)
            Y += [label] * lenx  # 在列表 Y 中 加入 列表 X 中对应样本的 标签---[label] * lenx：将标签 label 在 Y 中重复 lenx 次
            label += 1
        return X, Y

    def one_hot(y_train, y_valid_test):
        '''
        函数说明：one-hot编码
                    将样本标签编码为 含有 10个 元素的列表：[1，0，0，0，0，0，0，0，0，0]
                    分别对应文件夹中的十个文件，如，第一个文件是一个 故障数据 的文件，其故障对应到列表的第一个元素（第一个元素为1）
                    所以，这就是一个10分类问题，即最终要找出故障的位置

        Parameters:
            y_train : 训练集标签
            y_valid_test : <验证和测试集>标签
        Returns:
            y_train : 编码后的训练集标签
            y_valid_test : 编码后的<验证和测试集>标签
        '''
        y_train = np.array(y_train).reshape([-1, 1])
        y_valid_test = np.array(y_valid_test).reshape([-1, 1])

        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(y_train)  # 因为 y_train 和 y_valid_test 的类别相同，所以在其中一个上面fit就可以了

        y_train = Encoder.transform(y_train).toarray()
        y_valid_test = Encoder.transform(y_valid_test).toarray()

        y_train = np.asarray(y_train, dtype=np.int32)
        y_valid_test = np.asarray(y_valid_test, dtype=np.int32)
        return y_train, y_valid_test

    def scalar_stand(X_train, X_valid_test):
        '''
        函数说明：用训练集标准差标准化训练集以及测试集

        Parameters:
            X_train : 训练集
            X_valid_test : 验证和测试集
        Returns:
            X_train : 标准化后的训练集
            X_valid_test : 标准化后的<验证和测试集>
        '''
        # TODO: 这里要将每个模型训练时的标准化数据跟着模型一起储存下来，否则，当实时诊断时，新输入的数据重新进行标准化，会造成标准化尺度不相同，
        #  使得诊断结果不准确，以后解决。或者换个标准化方法
        scalar = preprocessing.StandardScaler().fit(X_train)
        X_train = scalar.transform(X_train)
        X_valid_test = scalar.transform(X_valid_test)
        return X_train, X_valid_test

    def valid_test_slice(X_valid_test, y_valid_test):
        '''
        函数说明：划分 验证集 和 测试集

        Parameters:
            X_valid_test : <验证及测试集>
            y_valid_test : <验证及测试集>标签
        Returns:
            X_valid : 验证集
            y_valid : 验证集标签
            X_test : 测试集
            y_test : 测试集标签
        '''
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)  # 分层抽样，随机的按比例选取 验证集 和 测试集
        '''
        因为 StratifiedShuffleSplit()函数只能将数据一分为二，不能将数据一分为三，所以需要在前面先将数据分为 训练集 和 验证以及测试集，
        然后再使用该函数进一步的将 验证以及测试集 分为 验证集 和 训练集
        '''
        for valid_index, test_index in ss.split(X_valid_test, y_valid_test):
            X_valid, X_test = X_valid_test[valid_index], X_valid_test[test_index]
            y_valid, y_test = y_valid_test[valid_index], y_valid_test[test_index]
            return X_valid, y_valid, X_test, y_test

    # 从所有.mat文件中读取出数据的字典
    data_dict = capture()
    # 将数据切分为训练集、验证集及测试集
    train, valid_test = slice_enhance(data_dict)
    # 为训练集制作标签
    X_train, y_train = add_labels(train)
    # 为验证集及测试集制作标签
    X_valid_test, y_valid_test = add_labels(valid_test)
    '''
    print(y_train)
    print(y_valid_test)
    这两句程序的结果及分析见下方图片
    '''
    # 为所有数据集One-hot标签
    y_train, y_valid_test = one_hot(y_train, y_valid_test)
    # 数据 是否标准化.
    if normal:
        X_train, X_valid_test = scalar_stand(X_train, X_valid_test)
    else:  # 需要做一个数据转换，转换成np格式.
        X_train = np.asarray(X_train)
        X_valid_test = np.asarray(X_valid_test)
    # 将 验证及测试集 切分为 验证集 和 测试集
    X_valid, y_valid, X_test, y_test = valid_test_slice(X_valid_test, y_valid_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def diagnosis_stage_prepro(data_path, signal_length=864, signal_number=500, normal=True):
    '''
    诊断阶段对数据的预处理
    :param data_path: 数据路径
    :param signal_length: 信号长度
    :param signal_number: 信号数量
    :param normal: 是否标准化
    :return:
    '''
    file_name = data_path.split('/')[-1].split('.')[0]  # 获得文件名

    def capture():
        """
        函数说明：读取mat文件，并将数据以字典返回（文件名作为key，数据作为value）

        Parameters:
            无
        Returns:
            data_dict : 数据字典
        """
        data_dict = {}

        file = loadmat(data_path)  # 读取 .mat 文件，返回的是一个 字典
        file_keys = file.keys()  # 获得该字典所有的key
        for key in file_keys:  # 遍历key, 获得 DE 的数据
            if 'DE' in key:  # DE: 驱动端 振动数据
                data_dict[file_name] = file[key].ravel()
        return data_dict

    def slice(data_dict):
        """
        函数说明：切取数据样本

        Parameters：
            data_dict : dict, 要进行划分的数据
        Returns:
            diagnosis_samples_dict : 切分后的 诊断样本
        """
        diagnosis_samples_dict = {}  # 训练集 样本

        keys = data_dict.keys()
        for key in keys:
            slice_data = data_dict[key]  # 获得value，即取得该文件里的 DE 数据
        all_lenght = len(slice_data)  # 获得数据长度
        sample_number = int(signal_number)  # 需要采集的信号 个数，防止输入小数，所以将其转为int

        samples = []  # 该文件中抽取的样本
        for j in range(sample_number):  # 在该文件中 抽取 信号，共抽取sample_number个（随机抽取）
            random_start = np.random.randint(low=0, high=(all_lenght - signal_length))  # high=(all_lenght - signal_length)：保证从任何一个位置开始都可以取到完整的数据长度
            sample = slice_data[random_start: random_start + signal_length]  # 抽取信号
            samples.append(sample)

        diagnosis_samples_dict[key] = samples  # 字典存储---文件名：对应的信号
        return diagnosis_samples_dict

    def scalar_stand(X_train):
        '''
        函数说明：用训练集标准差标准化训练集

        Parameters:
            X_train : 训练集
            X_valid_test : 验证和测试集
        Returns:
            X_train : 标准化后的训练集
        '''
        scalar = preprocessing.StandardScaler().fit(X_train)
        X_train = scalar.transform(X_train)
        return X_train

    # 从.mat文件中读取出数据的字典
    data_dict = capture()
    # 将数据按样本要求切分
    diagnosis_samples_dict = slice(data_dict)
    # diagnosis_samples = []
    diagnosis_samples = diagnosis_samples_dict[file_name]  # 取得对应文件中样本的数据
    # diagnosis_samples += x

    # 数据 是否标准化.
    if normal:
        diagnosis_samples = scalar_stand(diagnosis_samples)
    else:  # 需要做一个数据转换，转换成np格式.
        diagnosis_samples = np.asarray(diagnosis_samples)

    return diagnosis_samples
