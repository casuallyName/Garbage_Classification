# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 19:43
# @Author: Casually
# @File: args.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
# 定义需要用到的参数

import argparse

# 创建参数的解析对象
parser = argparse.ArgumentParser(description='PyTorch garbage Training ')

class_id2name = {0: '其他垃圾（干垃圾）', 1: '厨余垃圾（湿垃圾）', 2: '可回收物', 3: '有害垃圾'}
model_list = ['ResNeXt_32x8d', 'ResNeXt_32x16d_wsl', 'ResNeXt_32x32d_wsl', 'MyNet', 'DenseNet121', 'AlexNet',
              'GoogleNet', 'VGG_16']
# 设置参数信息

parser.add_argument('--model_name', default='ResNeXt_32x16d_WSL_Attention', type=str, choices=model_list, help='选择训练使用模型')
parser.add_argument('--feature_extract', default=True, type=bool, choices=model_list, help='固定特征开关')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='学习率')
parser.add_argument('--batch_size', default=8, type=int, help=' 批大小')
parser.add_argument('--num_workers', default=2, type=int, help=' 多线程数')
parser.add_argument('--resume', default=None, type=str, help='最后一个checkpoint断点的路径，为空时从头开始')
parser.add_argument('--log_path', default="Log/ResNeXt_32x16d_WSL/Attention", type=str, help='日志存储路径')
parser.add_argument('--checkpoint_path', default="checkpoint/ResNeXt_32x16d_WSL/Attention", type=str, help='checkpoint断点存储路径')
parser.add_argument('--checkpoint_name', default="checkpoint.pth.tar", type=str, help='checkpoint断点存储路径')
parser.add_argument('--pth_path', default="pth_files/ResNeXt_32x16d_WSL/Attention", type=str, help='训练结果存储路径')
parser.add_argument('--pth_name', default="ResNeXt_32x16d_WSL_Attention.pth", type=str, help='训练结果存储名称')
parser.add_argument('--epochs', default=10, type=int, help='迭代次数')
parser.add_argument('--num_classes', default=4, type=int, help='图片分类数')
parser.add_argument('--class_id2name', default=class_id2name, help='图片分类标签映射')
parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adam'], help='模型优化器')

# 进行参数解析
args = parser.parse_args(args=[])