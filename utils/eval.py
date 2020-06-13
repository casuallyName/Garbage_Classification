# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:07
# @Author: Casually
# @File: eval.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#

def accuracy(output, target, topk=(1,)):
    '''
    计算acc
    :param output: 预测类别标签
    :param target:  验证集标签
    :param topk:
    :return:
    '''
    # 取top1准确率，默认取top1，若取top1和top5准确率改为max((1,5))
    maxk = max(topk)
    batch_size = target.size(0)
    # 计算Topk
    _, pred = output.topk(maxk, 1, True, True)
    # output：输入张量
    # maxk: “top-k”中的k
    # dim: 排序的维
    # largest(bool): 控制返回最大或最小值
    # sorted(bool): 控制返回值是否排序
    pred = pred.t()  # 矩阵转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # 经过target.view(1，-1)后变成一维，再经过expand_as(pred)，该函数代表将target.view()扩展到跟pred相同的维度
    # pred.eq() 输出最大值的索引位置，这个索引位置和真实值的索引位置比较相等的做统计，就是这个批次准确的个数
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)  # axis=1表示按行相加 , axis=0表示按列相加
        res.append(correct_k.mul_(100.0 / batch_size))  # 对应位置点乘，个数 ==> 准确率
    return res
