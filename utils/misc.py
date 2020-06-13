# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:21
# @Author: Casually
# @File: misc.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
# 一些杂类方法定义

import torch
import os, csv
from args import args

__all__ = ['AverageMeter', 'get_optimizer', 'save_checkpoint']


def get_optimizer(model, args):
    '''
    优化方式选择
    :param model:  模型
    :param args:  参数
    :return:
    '''
    if args.optimizer == 'SGD':  # SGD优化算法
        return torch.optim.SGD(model.parameters(), args.lr)
    elif args.optimizer == 'Adam':  # Adam优化算法
        return torch.optim.Adam(model.parameters(), args.lr)
    else:
        raise NotImplementedError



class AverageMeter(object):
    # 计算并存储平均值和当前值，使用类方法方便更新，来自Github官方例程
    # Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L359
    def __init__(self):
        self.reset()

    def reset(self):
        '''
        重置参数
        :return:
        '''
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        '''
        跟新数据
        :param val:
        :param n:
        :return:
        '''
        self.val = val
        self.sum += val * n  # 求和
        self.count += n  # 数据更新次数
        self.avg = self.sum / self.count  # 求平均


def save_checkpoint(state, is_best, checkpoint=args.checkpoint_path, pth_files=args.pth_path,
                    checkpoint_name=args.checkpoint_name, pth_name=args.pth_name):
    '''
    保存断点信息
    :param state: 参数
    :param is_best:  是否是最优模型
    :param checkpoint:  断点路径
    :param pth_files: 模型保存路径
    :param checkpoint_name:  断点保存名称
    :param pth_name:  模型保存名称
    :return:
    '''
    # 检查路径是否存在/创建路径
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    if not os.path.exists(pth_files):
        os.makedirs(pth_files)
    # 保存断点信息
    filepath = os.path.join(checkpoint, checkpoint_name)
    torch.save(state, filepath)
    # 模型保存
    if is_best:
        model_path = os.path.join(pth_files, pth_name)
        torch.save(state['state_dict'], model_path)
        return True
    else:
        return False
