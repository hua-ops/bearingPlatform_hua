#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/2 17:34

@Author: Sun Jiahua

@File  : message_signal.py

@Desc  : 自定义的 信号类
"""

from PySide2.QtCore import QObject, Signal


class MyMessageSignal(QObject):
    send_msg = Signal(dict)

    # def run(self):
    #     self.send_msg.emit('发送信号')
