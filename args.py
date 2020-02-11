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

# 设置参数信息

# 1. 模型名称
# 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
# 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
# 'resnext101_32x16d_wsl'
parser.add_argument('--model_name', default='resnext101_32x16d_wsl', type=str,
                    choices=['resnext101_32x8d', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl'],
                    help='选择训练使用模型')

# 2. 学习率
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='学习率 1e-2,12-4,0.001')

# 3. 模型评估 默认false,指定 －e true
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='模型评估开关')

# 4. 批量数据加载尺寸
parser.add_argument('--batch_size', default=12, type=int, metavar='SIZE', help=' 加载包大小')
parser.add_argument('--num_workers', default=0, type=int, metavar='NUM', help=' 线程数')

# 5. 模型的存储路径

parser.add_argument('--resume', default="checkpoint/checkpoint.pth.tar", type=str, help='最后一个checkpoint断点的路径')
# parser.add_argument('--resume', default="", type=str, help='最后一个checkpoint断点的路径')
parser.add_argument('-c', '--checkpoint', default="checkpoint", type=str, help='checkpoint断点存储路径')
parser.add_argument('--checkpoint_name', default="checkpoint.pth.tar", type=str, help='checkpoint断点存储路径')
parser.add_argument('--pth_path', default="pth_files", type=str, help='训练结果存储路径')
parser.add_argument('--pth_name', default="GarbageClassificationOnResNeXt101.pth", type=str, help='训练结果存储名称')

# 6. 模型迭代次数
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='迭代次数')

# 7. 图片分类
parser.add_argument('--num_classes', default=4, type=int, metavar='N', help='图片分类数')
class_id2name = {0: '其他垃圾', 1: '厨余垃圾', 2: '可回收物', 3: '有害垃圾'}
parser.add_argument('--class_id2name', default=class_id2name, help='图片分类标签映射')


# 8. 开始位置（从哪个epoch开始训练）
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='Epoch开始位置')

# 9. 模型优化器
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], metavar='N',
                    help='optimizer(default adam)')

# 进行参数解析
args = parser.parse_args()
