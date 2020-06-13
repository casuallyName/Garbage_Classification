# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/3/26 下午7:22
# @Author: Casually
# @File: my_model.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional

all = ['My_Net']


class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.ReLU2 = nn.ReLU(inplace=True)
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.ReLU3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.ReLU4 = nn.ReLU(inplace=True)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.Avd_Popl = nn.AdaptiveAvgPool2d((7, 7))
        self.FC1 = nn.Linear(6272, 512)
        self.FC2 = nn.Linear(in_features=512, out_features=64)
        self.FC3 = nn.Linear(in_features=64, out_features=4)


    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.Pool1(x)
        x = self.conv3(x)
        x = self.ReLU3(x)
        x = self.conv4(x)
        x = self.ReLU4(x)
        x = self.Pool2(x)
        x = self.Avd_Popl(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x



