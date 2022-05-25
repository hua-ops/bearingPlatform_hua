# bearingPlatform_hua
西储大学轴承数据集故障诊断的仿真平台

## 1.简介
因为学习需要，因而简单学习了一下PySide2的使用，并粗暴的开发一款简单的故障诊断仿真平台（真的是简单粗暴的一个平台￣□￣｜｜)
），该平台使用[西储大学轴承数据集](https://www.cnblogs.com/gshang/p/10712809.html)实现了对轴承的故障诊断。平台主要功能： 

* 选择不同算法训练模型
* 使用保存的模型进行故障诊断
## 2.环境
* Windows 10
* python 3.6
* anaconda
* Pycharm
## 3.框架与依赖
* keras 2.24
* tensorflow-gpu 1.12
* pyside2 5.15.2
* skiit-learn 0.23
* numpy
* pandas
* matplotlib
## 4.详细说明
## 4.1 文件说明
UI 存放的软件平台页面布局文件

data_preprocess.py 数据预处理

diagnosis.py 故障诊断相关函数

feature_extraction.py 特征提取函数

main.py 主程序

message_signal.py 自定义信号

preprocess_train_result.py 处理模性训练结果的相关函数

training_model.py 模型训练的相关函数
### 4.1 故障分类算法
算法可以对**0马力，采样频率为48KHZ**的轴承的9类故障以及正常状态进行分类，这9类故障分别为：
* 滚动体故障：0.1778mm
* 滚动体故障：0.3556mm
* 滚动体故障：0.5334mm
* 内圈故障：0.1778mm
* 内圈故障：0.3556mm
* 内圈故障：0.5334mm
* 外圈故障（6点方向）：0.1778mm
* 外圈故障（6点方向）：0.3556mm
* 外圈故障（6点方向）：0.5334mm

平台中一共使用了4种不同的算法来进行故障诊断，这4种算法分别为：
* 1D_CNN
* LSTM
* GRU
* 随机森林

对于故障诊断的算法以及数据的处理，参考了[Jiali Zhang](https://github.com/zhangjiali1201/keras_bearing_fault_diagnosis)的代码。

*对于整体的算法可能并不是很完美，欢迎大家一起讨论改善*

## 5.效果图

<img src="https://github.com/hua-ops/bearingPlatform_hua/blob/master/UI/images/diagnosis_page.jpg" alt="故障诊断页面" style="zoom: 67%;" />

<img src="https://github.com/hua-ops/bearingPlatform_hua/blob/master/UI/images/train_model_page.jpg" alt="训练模型页面" style="zoom: 67%;" />

## 6.改进
这里的显示图片是将其先存到本地，然后再读取显示。后期在新项目中将其改进为使用 GraphicsView控件 嵌入Matplotlib的绘图，但因为这个新项目的诊断算法不太方便透露，所以大家可以参考我的这篇[Pyside2中嵌入Matplotlib的绘图](https://blog.csdn.net/qq_28053421/article/details/113828372?spm=1001.2014.3001.5501)，或者直接与我讨论交流！！！
