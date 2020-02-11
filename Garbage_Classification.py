# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 19:26
# @Author: Casually
# @File: Garbage_Classification.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm


import os, time
import torch
import torch.nn as nn
from torchvision import datasets
from args import args  # 参数定义
from utils import transform
from utils.model import train, evaluate, initital_model
from utils.logger import Logger
from utils.misc import save_checkpoint, get_optimizer  # 模型保存、优化器
from utils import Mail
from sklearn import metrics  # 训练矩阵效果评估工具类
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 0. 数据划分
key = False  # 数据划分
if key:
    All_Data_Path = 'data/'
    import Data_Preprocess as dp

    img_path_list, img_name2label_dict, img_label2count_dict = dp.get_img_info(All_Data_Path)
    train_json_path, verify_json_path, train_img_list, verify_img_list = dp.data_shuffle_split(0.8, img_path_list)

# 1. 数据整体探测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('{} 设备类型：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), str(device).upper()))
Data_Path = 'data/train_data'
Log_Path = 'Log'
if not os.path.exists(Log_Path):
    os.makedirs(Log_Path)

# 2. 数据封装ImageFolder格式
TRAIN = "{}/train".format(Data_Path)
VALID = "{}/val".format(Data_Path)

print('{} 数据封装：\t'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
train_data = datasets.ImageFolder(root=TRAIN, transform=transform.preprocess)
val_data = datasets.ImageFolder(root=VALID, transform=transform.preprocess)
print('完成')

print('{} 校验数据：\t'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
assert train_data.class_to_idx.keys() == val_data.class_to_idx.keys()
print('完成')

# 3. 批量数据加载
print('{Time} 设定\t包大小：{batch_size} | 线程数：{num_workers}'.format(
    Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
    batch_size=args.batch_size,
    num_workers=args.num_workers
))

print('{} 装载数据：\t'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False)
print('完成')
'''
# torch.utils.data.DataLoader(dataset, 
#                             batch_size=1,   # 每次加载batch_size张图片
#                             shuffle=False,  # 打乱顺序(True/False)
#                             sampler=None,
#                             batch_sampler=None,
#                             num_workers=0,  # 多线程输入线程数
#                             collate_fn=None,
#                             pin_memory=False, 
#                             drop_last=False, t
#                             imeout=0,
#                             worker_init_fn=None)
# image,label = next(iter(train_loader))
#
# print(label)
# # ([10, 3, 224, 224])
# # 十张图片，3通道，224x224
# print(image.shape)
'''

# 所有分类['其他垃圾', '厨余垃圾', '可回收物', '有害垃圾']
class_list = args.class_id2name
# [class_id2name()[i] for i in list(range(len(train_data.class_to_idx.keys())))]
# 定义全局变量，保存准确率
best_acc = 0


# 4. 定义模型训练
def run(model, train_loader, val_loader):
    '''
    模型训练和预测
    :param model: 初始化的model
    :param train_loader: 训练数据
    :param val_loader: 验证数据
    :return:
    '''

    # 模型保存的变量
    global best_acc

    # 训练C类别的分类问题，用CrossEntropyLoss调整权重
    criterion = nn.CrossEntropyLoss()
    # torch.optim 是一个各种优化算法库
    # 得到optimizer 对象，能保存当前的参数状态并且基于计算梯度更新参数
    optimizer = get_optimizer(model, args)
    # 断点续传
    if args.resume:
        print('{} 正在恢复断点...'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())))
        if not os.path.exists(args.resume):
            raise FileNotFoundError('No such file or directory: {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = args.start_epoch
    # 评估: 混淆矩阵、准确率、召回率、F1-score
    if args.evaluate:
        print('\nEvaluate only')
        test_loss, test_acc, predict_all, labels_all = evaluate(val_loader, model, criterion, test=True)
        print('Test Loss:%.8f,Test Acc:%.2f' % (test_loss, test_acc))

        # 混淆矩阵 的数据处理
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)

        print('\n report ', report)
        print('\n confusion', confusion)
        return

    # 添加日志文件
    logname = '{}.txt'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
    logger = Logger(os.path.join(Log_Path, logname), title=None)
    # 设置logger 的头信息
    logger.set_names(['epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc', 'Valid Acc', 'Save'])

    # start_epoch 断点续传使用
    for epoch in range(start_epoch, args.epochs + 1):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer,epoch, args.epochs)
        # val
        test_loss, test_acc = evaluate(val_loader, model, criterion, epoch, args.epochs, test=None)
        # 保存模型
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        res = save_checkpoint({
            'epoch': epoch + 1,  # 迭代此次数
            'state_dict': model.state_dict(),  # 状态
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()

        }, is_best, checkpoint=args.checkpoint, pth_files=args.pth_path)
        # 核心参数保存logger
        logger.append([epoch, args.lr, train_loss, test_loss, train_acc, test_acc, res])

        res_mail = Mail.mail(
            to_user='fjklqq@163.com',
            subject='第{}轮训练结果'.format(epoch),
            info='训练损失：{T_loss:.4f}\n验证损失：{V_loss:.4f}\n训练准确度：{T_acc:.4f}\n验证准确度：{V_acc:.4f}\n保存模型：{RES}'.format(
                T_loss=train_loss,
                V_loss=test_loss,
                T_acc=train_acc,
                V_acc=test_acc,
                RES=res)
        )
        print(
            '{Time} [{epoch}/{epochs}] | 训练损失：{T_loss:.4f} | 验证损失：{V_loss:.4f} | 训练准确度：{T_acc:.4f} | 验证准确度：{V_acc:.4f} | 保存模型：{RES} | 邮件发送：{RES_mail}'.format(
                Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
                epoch=epoch,
                epochs=args.epochs,
                T_loss=train_loss,
                V_loss=test_loss,
                T_acc=train_acc,
                V_acc=test_acc,
                RES=res,
                RES_mail=res_mail))

    print('{} 最高准确度：\t{:.2f}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), best_acc))


# 入口程序
if __name__ == '__main__':
    # 模型初始化
    model_name = args.model_name
    num_classes = args.num_classes
    model_ft = initital_model(model_name, num_classes, feature_extract=True)
    model_ft.to(device)  # 设置模型运行模式（cuda／cpu)
    run(model_ft, train_loader, val_loader)