# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:21
# @Author: Casually
# @File: misc.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
# Some helper functions for PyTorch, including:

import torch
import os, shutil
from args import args

__all__ = ['AverageMeter', 'get_optimizer', 'save_checkpoint']


# Checkpoint = args.checkpoint
# Pth_files = args.pth_files
# Filename = args.checkpoint_name


def get_optimizer(model, args):
    '''
    优化方式选择
    :param model:  模型
    :param args:  参数
    :return:
    '''
    if args.optimizer == 'sgd':
        # 梯度下降
        return torch.optim.SGD(model.parameters(),
                               args.lr)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(),
                                   args.lr)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(),
                                args.lr)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint=args.checkpoint, pth_files=args.pth_path,
                    checkpoint_name=args.checkpoint_name, pth_name=args.pth_name):
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

