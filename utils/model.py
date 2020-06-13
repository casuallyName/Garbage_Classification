# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:03
# @Author: Casually
# @File: model.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
# 一些与模型有关方法


import time
import torch
import torch.nn as nn
import models
from utils.eval import accuracy
from utils.misc import AverageMeter
from utils import ProgressBar as Bar
import numpy

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
    eta_time = AverageMeter()  # eta时间
    batch_time = AverageMeter()  # 批次时间
    losses = AverageMeter()  # 损失
    top1 = AverageMeter()  # 准确率
    end = time.time()  # 计算eta时间

    model.train()  # 训练模式

    # 进行模型的训练
    for batch_index, (inputs, targets) in enumerate(train_loader):
        now_time = time.time()
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

        # 计算acc和变量更新,这里只拿了top1
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))  # 更新损失
        top1.update(prec1.item(), inputs.size(0))  # 更新top1
        batch_time.update(time.time() - end)  # 更新批次时间
        end = time.time()  # 更新概论结束时间
        eta_time.update(now_time + (end - now_time) * (len(train_loader) - batch_index - 1))  # 更新eta时间
        # 进度条
        Bar.show(
            msg_behind='[{epoch}/{epochs}] TrainEpoch: '.format(
                epoch=epoch,  # 当前epoch
                epochs=epochs  # 全部epoch
            ),
            batch=batch_index,  # 当前batch
            size=len(train_loader),  # 全部batch
            msg_front=' | Batch: {bt:.4f}s | ETA: {estimate} | Loss: {loss:.4f} | Top1: {top1: .4f}'.format(
                bt=batch_time.val,  # 批次用时
                estimate=time.strftime("[%Y.%m.%d %H:%M:%S]", time.localtime(eta_time.val)),  # ETA
                loss=losses.avg,  # 当前损失
                top1=top1.avg  # 当前准确率
            ))

    return (losses.avg, top1.avg)


def evaluate(val_loader, model, criterion, epoch='', epochs=''):
    '''
    模型评估
    :param val_loader:
    :param model:
    :param criterion:
    :param epoch:
    :param epochs:
    :param test:
    :return:
    '''
    global best_acc

    batch_time = AverageMeter()  # 初始化batch时间
    eta_time = AverageMeter()  # 初始化eta时间
    losses = AverageMeter()  # 初始化损失
    top1 = AverageMeter()  # 初始化准确率


    # predict_all = numpy.array([], dtype=int)
    # labels_all = numpy.array([], dtype=int)

    model.eval()
    end = time.time()

    # 训练每批数据，然后进行模型的训练
    for batch_index, (inputs, targets) in enumerate(val_loader):
        now_time = time.time()
        # move tensors to GPU if cuda is_available
        inputs, targets = inputs.to(device), targets.to(device)
        # 模型的预测
        outputs = model(inputs)
        # 计算loss
        loss = criterion(outputs, targets)

        # 计算acc和变量更新,这里只拿了top1
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        batch_time.update(time.time() - end)

        # 评估混淆矩阵的数据
        # targets = targets.data.cpu().numpy()  # 真实数据的y数值
        # predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 预测数据y数值
        # labels_all = numpy.append(labels_all, targets)  # 数据赋值
        # predict_all = numpy.append(predict_all, predic)

        end = time.time()
        eta_time.update(now_time + (end - now_time) * (len(val_loader) - batch_index - 1))  # 更新eta时间


        # 进度条
        Bar.show(
            msg_behind='[{epoch}/{epochs}] EvaluateEpoch: '.format(epoch=epoch, epochs=epochs),
            batch=batch_index,
            size=len(val_loader),
            msg_front=' | Batch: {bt:.4f}s  | ETA: {estimate} | Loss: {loss:.4f} | Top1: {top1: .4f}'.format(
                bt=batch_time.val,
                estimate=time.strftime("[%Y.%m.%d %H:%M:%S]", time.localtime(eta_time.val)),
                loss=losses.avg,
                top1=top1.avg
            ))
    else:  # 数据校验时使用
        return (losses.avg, top1.avg)


def set_parameter_requires_grad(model, feature_extract):
    '''
    :param model:  模型
    :param feature_extract: true 固定特征抽取层
    :return:
    '''
    if feature_extract:
        # 遍历每个参数，并冻结参数
        for param in model.parameters():
            param.requires_grad = False


def init_model(model_name, num_classes, feature_extract=True):
    '''
    初始化模型
    :param model_name: 提供的模型名称
    :param num_classes: 图片分类个数
    :param feature_extract: 设置true ，固定特征提取层，优化全连接的分类器
    :return:
    '''
    print('{} 模型初始化：\t'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
    if model_name == 'ResNeXt_32x16d_wsl':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x16d_wsl()
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    elif model_name == 'ResNeXt_32x8d':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x8d()
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    elif model_name == 'ResNeXt_32x32d_wsl':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x32d_wsl()
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs, out_features=num_classes)
        )
    elif model_name == 'DenseNet121':
        print(model_name)
        # 初始化模型
        model_ft = models.densenet161(feature_extract)
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 修改FC层的分类个数
        model_ft.classifier = nn.Linear(2208, num_classes, bias=True)
    elif model_name == 'AlexNet':  # need 1 hour
        print(model_name)
        # 初始化模型
        model_ft = models.alexnet(pretrained=True)
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 修改FC层的分类个数
        model_ft.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    elif model_name == 'GoogleNet':
        print(model_name)
        # 初始化模型
        model_ft = models.inception_v3(feature_extract)
        model_ft.aux_logits = False  # error：   AttributeError: 'tuple' object has no attribute 'log_softmax'
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=2048, out_features=num_classes)
        )
    elif model_name == 'VGG_16':
        print(model_name)
        # 初始化模型
        model_ft = models.vgg16(feature_extract)
        # 设置 固定特征提取层
        set_parameter_requires_grad(model_ft, feature_extract)
        # 修改FC层的分类个数
        model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif model_name == 'ResNeXt_32x8d_WSL_Attention':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x8d_attention(pretrained = feature_extract)

        # 设置 固定特征提取层
        for k,v in model_ft.named_parameters():
            if not ('ca1' in k or 'sa1' in k or 'ca2' in k or 'sa2' in k):
                v.requires_grad = False
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs,out_features=num_classes)#1024),
        )
    elif model_name == 'ResNeXt_32x16d_WSL_Attention':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x16d_wsl_attention(pretrained = feature_extract)

        # 设置 固定特征提取层
        for k,v in model_ft.named_parameters():
            if not ('ca1' in k or 'sa1' in k or 'ca2' in k or 'sa2' in k):
                v.requires_grad = False
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs,out_features=num_classes)#1024),
        )
    elif model_name == 'ResNeXt_32x32d_WSL_Attention':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x32d_wsl_attention(pretrained = feature_extract)

        # 设置 固定特征提取层
        for k,v in model_ft.named_parameters():
            if not ('ca1' in k or 'sa1' in k or 'ca2' in k or 'sa2' in k):
                v.requires_grad = False
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs,out_features=num_classes)#1024),
        ) 
    elif model_name == 'ResNeXt_32x48d_WSL_Attention':
        print(model_name)
        # 初始化模型
        model_ft = models.resnext101_32x48d_wsl_attention(pretrained = feature_extract)

        # 设置 固定特征提取层
        for k,v in model_ft.named_parameters():
            if not ('ca1' in k or 'sa1' in k or 'ca2' in k or 'sa2' in k):
                v.requires_grad = False
        # 拿到FC层原有的输入
        num_ftrs = model_ft.fc.in_features
        # 修改FC层的分类个数
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(in_features=num_ftrs,out_features=num_classes)#1024),
        ) 

    elif model_name == 'MyNet':
        print(model_name)
        # 初始化模型
        model_ft = models.My_Net()

    else:
        print('失败')
        raise TypeError('Invalid model name,exiting')

    return model_ft
