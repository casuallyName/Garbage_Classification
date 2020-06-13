# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 19:26
# @Author: Casually
# @File: main.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm


import os, time
import torch
import torch.nn as nn
import torch.backends.cudnn
from torchvision import datasets
from args import args  # 参数定义
from utils import transform
from utils.model import train, evaluate, init_model
from utils.logger import Logger
from utils.misc import save_checkpoint, get_optimizer  # 模型保存、优化器
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# IOError: image file is truncated (5 bytes not processed)
# PIL无法读取有数据缺失的图片（网络图片下载是可能会缺少有些信息），可以通过一个设置使其接受这一点

torch.backends.cudnn.benchmark = True  # cuDNN加速


def Data_Divide(All_Data_Path):
    '''
    # 数据划分
    :param All_Data_Path:
    :return:
    '''
    import data_preprocess as dp
    img_path_list, _, _ = dp.get_img_info(All_Data_Path)  # 取得路径下所有图片的相对路径
    dp.data_shuffle_split(0.8, img_path_list)  # 划分训练集与验证集


def Data_Folder(Data_Path):
    '''
    数据封装ImageFolder格式
    :param Data_Path:
    :return:
    '''
    TRAIN = "{}/train".format(Data_Path)  # 训练数据路径
    VALID = "{}/val".format(Data_Path)  # 验证数据路径
    train_data = datasets.ImageFolder(root=TRAIN, transform=transform.preprocess)  # 训练数据封装ImageFolder形式
    val_data = datasets.ImageFolder(root=VALID, transform=transform.preprocess)  # 验证数据封装ImageFolder形式

    assert train_data.class_to_idx.keys() == val_data.class_to_idx.keys()  # 验证训练集与验证集标签数

    return train_data, val_data


def Data_Loader(train_data, val_data, batch_size=args.batch_size, num_workers=args.num_workers):
    '''
    批量数据加载
    :param train_data:
    :param val_data:
    :param args:
    :return:
    '''
    train_loader = torch.utils.data.DataLoader(
        train_data,  # 训练数据集
        batch_size=batch_size,  # 批次大小
        num_workers=num_workers,  # 多线程加载数据
        shuffle=True  # 打乱数据
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,  # 验证数据集9
        batch_size=batch_size,  # 批次大小
        num_workers=num_workers,  # 多线程加载数据
        shuffle=False  # 不打乱数据
    )

    return train_loader, val_loader


def Train_Run(model, train_loader, val_loader):
    '''
        模型训练和预测
    :param model: 初始化的model
    :param train_loader: 训练数据
    :param val_loader: 验证数据
    :param log_path:  日志路径
    :return:
    '''

    best_acc = 0

    # 损失函数，用CrossEntropyLoss调整权重
    criterion = nn.CrossEntropyLoss()

    # 得到optimizer 对象，能保存当前的参数状态并且基于计算梯度更新参数
    optimizer = get_optimizer(model, args)

    # 断点续传
    if args.resume:
        print('{} 正在恢复断点...'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())))
        if not os.path.exists(args.resume):
            raise FileNotFoundError('No such file or directory: {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')  # 先把数据加载到内存里
        best_acc = checkpoint['best_acc']  # 更新最高准确率
        start_epoch = checkpoint['epoch']  # 更新epoch
        model.load_state_dict(checkpoint['state_dict'])  # 更新模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 更新模型优化器
    else:
        start_epoch = 1

        # 添加日志文件
    logname = '{}.txt'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
    logger = Logger(os.path.join(args.log_path, logname), title=None)  # 实例化一个Log对象
    # 设置logger 的头信息
    logger.set_names(['epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc', 'Valid Acc', 'Save'])

    for epoch in range(start_epoch, args.epochs + 1):
        # 训练
        train_loss, train_acc = train(train_loader,  # 训练数据(Data_Loader)
                                      model,  # 模型(class)
                                      criterion,  # 损失函数
                                      optimizer,  # 优化器
                                      epoch,  # 当前epoch
                                      args.epochs  # 全部epoch
                                      )
        # 验证
        test_loss, test_acc = evaluate(val_loader,  # 验证数据(Data_Loader)
                                       model,  # 模型(class)
                                       criterion,  # 损失函数
                                       epoch,  # 当前epoch
                                       args.epochs  # 全部epoch
                                       )
        # 保存模型
        is_best = test_acc > best_acc  # 判断模型优劣
        best_acc = max(test_acc, best_acc)  # 取得最高准确率
        res = save_checkpoint(
            {
                'epoch': epoch + 1,  # 迭代此次数
                'state_dict': model.state_dict(),  # 模型参数
                'train_acc': train_acc,  # 当前训练准确率
                'test_acc': test_acc,  # 当前验证准确率
                'best_acc': best_acc,  # 最高验证准确率
                'optimizer': optimizer.state_dict()  # 模型优化器
            },
            is_best,  # 是否保存模型
            checkpoint=args.checkpoint_path,  # 断点信息保存路径
            pth_files=args.pth_path,  # 模型存储路径
            checkpoint_name=args.checkpoint_name,
            pth_name=args.pth_name
        )

        # 核心参数更新到日志
        logger.append([epoch, args.lr, train_loss, test_loss, train_acc, test_acc, res])
        print(
            '{Time} [{epoch}/{epochs}] | 训练损失：{T_loss:.4f} | 验证损失：{V_loss:.4f} | 训练准确度：{T_acc:.4f} | 验证准确度：{V_acc:.4f} | 保存模型：{RES}'.format(
                Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
                epoch=epoch,
                epochs=args.epochs,
                T_loss=train_loss,
                V_loss=test_loss,
                T_acc=train_acc,
                V_acc=test_acc,
                RES=res, ))
        # print(model.sa1.conv1.parameters)

    print('{} 最高准确度：\t{:.2f}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), best_acc))


if __name__ == '__main__':
    Data_Path = './data/train_data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 检查路径是否存在/创建路径
    if not os.path.exists(args.pth_path):
        os.makedirs(args.pth_path)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    print('{} 设备类型：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), str(device).upper()))
    print('{Time} 设定\t包大小：{batch_size} | 线程数：{num_workers}'.format(
        Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    ))
    print('{} 固定特征：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), args.feature_extract))

    # 模型初始化
    model_ft = init_model(args.model_name, args.num_classes, args.feature_extract)  # 初始化模型
    model_ft.to(device)
    train_data, val_data = Data_Folder(Data_Path)  # 封装数据
    train_loader, val_loader = Data_Loader(train_data, val_data)  # 加载数据
    Train_Run(model=model_ft,
              train_loader=train_loader,
              val_loader=val_loader
              )  # 训练
