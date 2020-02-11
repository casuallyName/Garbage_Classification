# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:03
# @Author: Casually
# @File: model.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm


import time
import torch
import torch.nn as nn

# 导入模型定义方法
import models

# 导入工具类
from utils.eval import accuracy
from utils.misc import AverageMeter
from utils import ProgressBar as Bar
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, criterion, optimizer, epoch='', epochs=''):
    '''
    模型训练
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param epochs:
    :return:
    '''
    # 定义保存更新变量
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()




    # 进行模型的训练
    for batch_index, (inputs, targets) in enumerate(train_loader):
        now_time = time.time()
        data_time.update(time.time() - end)
        # 如果有支持CUDA,就把 tensors 数据放大 GPU
        inputs, targets = inputs.to(device), targets.to(device)
        # 反向传播之前，用 zero_grad 方法清空梯度
        optimizer.zero_grad()
        # 模型的预测
        outputs = model(inputs)
        # 计算loss
        loss = criterion(outputs, targets)
        # 反向传播
        loss.backward()
        # 执行单个优化步骤（参数更新）
        optimizer.step()

        # 计算acc和变量更新
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        # 进度条
        Bar.show(
            msg_behind='[{epoch}/{epochs}] TrainEpoch: '.format(epoch=epoch, epochs=epochs),
            batch=batch_index,
            size=len(train_loader),
            msg_front=' | Data: {data:.4f}ms | Batch: {bt:.4f}s | Loss: {loss:.4f} | Top: {top1: .4f} | 本轮预计完成时间：{estimate}'.format(
                data=data_time.val * 1000,
                bt=batch_time.val,
                loss=losses.avg,
                top1=top1.avg,
                estimate=time.strftime("[%Y.%m.%d %H:%M:%S]", time.localtime(
                    now_time + (end - now_time) * (len(train_loader) - batch_index - 1))),
            ))

    return (losses.avg, top1.avg)


def evaluate(val_loader, model, criterion, epoch='', epochs='', test=None):
    '''
     模型评估
    :param val_loader:
    :param model:
    :param criterion:
    :param test:
    :param epoch:
    :param epochs:
    :return:
    '''
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model.eval()
    end = time.time()

    # 训练每批数据，然后进行模型的训练
    for batch_index, (inputs, targets) in enumerate(val_loader):
        now_time = time.time()
        data_time.update(time.time() - end)
        # move tensors to GPU if cuda is_available
        inputs, targets = inputs.to(device), targets.to(device)
        # 模型的预测
        outputs = model(inputs)
        # 计算loss
        loss = criterion(outputs, targets)

        # 计算acc和变量更新
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # 评估混淆矩阵的数据
        targets = targets.data.cpu().numpy()  # 真实数据的y数值
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 预测数据y数值
        labels_all = np.append(labels_all, targets)  # 数据赋值
        predict_all = np.append(predict_all, predic)

        # 进度条
        Bar.show(
            msg_behind='[{epoch}/{epochs}] EvaluateEpoch: '.format(epoch=epoch, epochs=epochs),
            batch=batch_index,
            size=len(val_loader),
            msg_front=' | Data: {data:.4f}ms | Batch: {bt:.4f}s | Loss: {loss:.4f} | Top: {top1: .4f}  | 本轮预计完成时间：{estimate}'.format(
                data=data_time.val * 1000,
                bt=batch_time.val,
                loss=losses.avg,
                top1=top1.avg,
                estimate=time.strftime("[%Y.%m.%d %H:%M:%S]", time.localtime(
                    now_time + (end - now_time) * (len(val_loader) - batch_index - 1)))
            ))
    if test:  # 数据训练时验证使用
        return (losses.avg, top1.avg, predict_all, labels_all)
    else:  # 数据校验时使用
        return (losses.avg, top1.avg)


def set_parameter_requires_grad(model, feature_extract):
    '''

    :param model:  模型
    :param feature_extract: true 固定特征抽取层
    :return:
    '''
    if feature_extract:
        for param in model.parameters():
            # 不需要更新梯度，冻结某些层的梯度
            param.requires_grad = False


def initital_model(model_name, num_classes, feature_extract=True):
    """
    基于提供的 pre_trained_model 进行初始化
    :param model_name: 提供的模型名称
    :param num_classes: 图片分类个数
    :param feature_extract: 设置true ，固定特征提取层，优化全连接的分类器
    :return:
    """
    print('{} 模型初始化：\t'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
    if model_name == 'resnext101_32x16d_wsl':
        print(model_name, end='')
        # 加载facebook pre_trained_model resnext101,默认1000 类
        model_ft = models.resnext101_32x16d_wsl()
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)

        # 调整分类个数
        num_ftrs = model_ft.fc.in_features
        # 修改fc 的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
        print('\t完成')
    elif model_name == 'resnext101_32x8d':
        print(model_name, end='')
        # 加载facebook pre_trained_model resnext101,默认1000 类
        model_ft = models.resnext101_32x8d()
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)

        # 调整分类个数
        num_ftrs = model_ft.fc.in_features
        # 修改fc 的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
        print('\t完成')
    elif model_name == 'resnext101_32x32d_wsl':
        print(model_name, end='')
        # 加载facebook pre_trained_model resnext101,默认1000 类
        model_ft = models.resnext101_32x32d_wsl()
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)

        # 调整分类个数
        num_ftrs = model_ft.fc.in_features
        # 修改fc 的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
        print('\t完成')
    else:
        print('失败')
        raise TypeError('Invalid model name,exiting')

    return model_ft



# import codecs
#
#
# def class_id2name():
#     '''
#     标签关系映射
#     :return:
#     '''
#
#     clz_id2name = {}
#
#     for line in codecs.open('utils/garbage_label.txt', 'r', encoding='utf-8'):
#         line = line.strip()
#         _id = line.split(":")[0]
#         _name = line.split(":")[1]
#         clz_id2name[int(_id)] = _name
#     return clz_id2name

