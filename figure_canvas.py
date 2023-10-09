#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/9/27 16:04

@Author: Sun Jiahua

@File  : figure_canvas.py

@Desc  : None
"""
from PySide2.QtWidgets import QGraphicsScene
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")  # 声明使用QT5


class MyFigureCanvas(FigureCanvas):
    """
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    """

    def __init__(self, graphics_view, parent=None, width=10, height=5, dpi=100):
        # 创建一个Figure
        fig = plt.Figure(figsize=(width, height), dpi=dpi, tight_layout=True)  # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)

        self.axes = fig.add_subplot(111)  # 添加子图
        self.axes.spines['top'].set_visible(False)  # 去掉绘图时上面的横线
        self.axes.spines['right'].set_visible(False)  # 去掉绘图时右面的横线
        self.axes.plot(range(-1, 2), [0, 0, 0])

        self.graphics_view = graphics_view
        # 加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.graphic_scene = QGraphicsScene()
        # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.graphic_scene.addWidget(self)
        self.graphics_view.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.graphics_view.show()  # 调用show方法呈现图形

    def plot(self, x, y, xlim=None, ylim=None, title=None):
        self.axes.clear()
        self.axes.plot(x, y)
        if xlim:
            self.axes.set_xlim(xlim)
        if ylim:
            self.axes.set_ylim(ylim)
        if title:
            self.axes.set_title(title)
        self.draw()
