# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/16 14:28
# @Author: Casually
# @File: DataPreprocess.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm

import json, random, os, sys, time
from glob import glob
from pyecharts import options as opts
from pyecharts.charts import Bar
from PIL import Image
import seaborn as sns  # 导入可视化库
import matplotlib.pyplot as plt
import numpy as np
import webbrowser  # 调用浏览器
import shutil
from collections import Counter  # 统计

import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
# from os import walk
from torch.backends import cudnn


# 整体探测
# class detection():
# 写入json
def __json_store(path, data):
    '''
    将字典写入json文件
    :param path: 存储目录
    :param data: 待写入数据
    :return:
    '''
    with open(path, 'w', ) as fw:
        json.dump(data, fw, indent=1, ensure_ascii=False)


# 加载json
def __json_load(path):
    '''
    读取json文件
    :param path: 文件目录
    :return: dict
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data


# 取得该路径下所所有image的文件名、路径及标签
def get_img_info(path):
    '''
    取得该路径下所所有image的文件名、路径及标签
    :param path:train_data path
    :return img_path_list : dict{'img_name', 'img_lable', 'img_path' }
    :return img_name2label_dict : dict{'img_name', 'img_lable' }
    :return img_label2count_dict : dict{'img_name', 'count' }
    '''

    img_path_list = []
    img_name2label_dict = {}
    img_label2count_dict = {}

    data_path_txt = os.path.join(path, 'All_data/*.txt')
    txt_file_list = glob(data_path_txt)  # 所有.txt文件的列表

    garbage_classify_rule = __json_load(os.path.join(path, 'garbage_classify_rule.json'))
    garbage_index_classify = __json_load(os.path.join(path, 'garbage_index_classify.json'))
    for item in garbage_classify_rule:  # 更新字典  40类更新为4类
        garbage_classify_rule[item] = garbage_index_classify[garbage_classify_rule[item].split('/')[0]]

    # 遍历文件列表
    for i, file in enumerate(txt_file_list):
        __processBar('{} 提取数据'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), i + 1, len(txt_file_list))
        with open(file, 'r') as f:
            line = f.readline().split(',')  # [0] img_name,[1]img_lable
        img_name2label_dict[line[0]] = line[1]
        img_path = os.path.join(path, 'All_data/{}'.format(line[0]))
        img_path_list.append({'img_name': line[0], 'img_lable': int(line[1]),
                              'img_4lable': int(garbage_classify_rule[line[1].strip()]), 'img_path': img_path})
        img_label2count_dict[int(line[1])] = img_label2count_dict.get(int(line[1]), 0) + 1

    img_label2count_dict = dict(
        sorted(img_label2count_dict.items(), key=lambda item: item[0]))  # 对img_label2count_dict内部按照Key进行从小到大排序
    # img_label2count_dict = dict(
    #   sorted(img_label2count_dict.items(), key=lambda item: item[1]))  # 对img_label2count_dict内部按照Value进行从小到大排序

    return img_path_list, img_name2label_dict, img_label2count_dict


# 取得该路径下所所有image的ID、尺寸、标签
def get_img_size(path):
    '''
    获取训练集中图片规格，返回一个包含（id, width, height, ratio, label）的列表
    :param path: train_data path
    :return : list(id, width, height, ratio, label)
    '''
    data = []
    _, img_name2label_dict, _ = get_img_info(path)
    data_path = os.path.join(path, 'All_data')

    img_file_path = os.path.join(data_path, '*.jpg')
    imgs_path = glob(img_file_path)

    for i, img_path in enumerate(imgs_path):
        __processBar('{} 提取图片信息'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), i + 1, len(imgs_path))
        img_id = img_path.split('_')[-1].split('.')[0]
        img_label = img_name2label_dict['img_{}.jpg'.format(img_id)]
        img = Image.open(img_path)
        data.append([int(img_id), img.size[0], img.size[1], float('{:.2f}'.format(img.size[0] / img.size[1])),
                     int(img_label)])

    return data


# 将train_data拆分为train集合verify集了两部分
def data_shuffle_split(size, img_path_list):
    '''
    将train_data拆分为train集合verify集了两部分
    :param size: 训练集所占比例
    :param img_path_list: get_img_info的一个return
    :return:
    '''
    # 原始数据进行随机排序清洗
    random.shuffle(img_path_list)
    # 设置数据分布分布比例 训练：验证 =  size：1-size
    if isinstance(size, float) and 0 < size < 1:
        # train_img_dict = {}
        # verify_img_dict = {}
        path = img_path_list[0]['img_path'].split(img_path_list[0]['img_name'])[0].split('All_data')[0]

        # 取得分界
        train_size = int(len(img_path_list) * size)
        # 截取训练集
        train_img_list = img_path_list[:train_size]
        # 4分类
        train_img_dict = dict([[item['img_path'], item['img_4lable']] for item in train_img_list])
        train_json_path = os.path.join(path, 'train_img_dict.json')
        __json_store(train_json_path, train_img_dict)
        # 40分类
        train_img_dict = dict([[item['img_path'], item['img_lable']] for item in train_img_list])
        train_json_path = os.path.join(path, 'train_img_dict_40.json')
        __json_store(train_json_path, train_img_dict)
        # 截取测试集
        verify_img_list = img_path_list[train_size:]
        # 4分类
        verify_img_dict = dict([[item['img_path'], item['img_4lable']] for item in verify_img_list])
        verify_json_path = os.path.join(path, 'verify_img_dict.json')
        __json_store(verify_json_path, verify_img_dict)
        # 40分类
        verify_img_dict = dict([[item['img_path'], item['img_lable']] for item in verify_img_list])
        verify_json_path = os.path.join(path, 'verify_img_dict_40.json')
        __json_store(verify_json_path, verify_img_dict)

        # with open(os.path.join(path,'train_img_list.txt'),'w') as f:
        #    for dict in train_img_list:
        #        f.write('{},{}\n'.format(dict['img_path'],dict['img_lable']))
        # with open(os.path.join(path, 'verify_img_list.txt'), 'w') as f:
        #    for dict in verify_img_list:
        #        f.write('{},{}\n'.format(dict['img_path'], dict['img_lable']))

        # 清空目录，避免函数重复调用时数据重复
        print('{} 清空目录数据: '.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
        shutil.rmtree(os.path.join(path, 'train_data'), ignore_errors=True)
        print('完成')

        # 图片－标签目录
        for i, item in enumerate(train_img_list):
            train_dir = os.path.join(path, 'train_data/train/{}'.format(item['img_4lable']))
            # 目录创建
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            # 图片数据进行拷贝
            shutil.copy(item['img_path'], train_dir)
            __processBar('{} 拷贝训练集'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), i + 1,
                         len(train_img_list))
        for i, item in enumerate(verify_img_list):
            verify_dir = os.path.join(path, 'train_data/val/{}'.format(item['img_4lable']))
            # 目录创建
            if not os.path.exists(verify_dir):
                os.makedirs(verify_dir)
            # 图片数据进行拷贝
            shutil.copy(item['img_path'], verify_dir)
            __processBar('{} 拷贝验证集'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), i + 1,
                         len(verify_img_list))

        return train_json_path, verify_json_path, train_img_list, verify_img_list
    else:
        raise TypeError('<size> 参数必须是在(0,1)上的浮点数')


# 数据可视化
def data_visualization(data_path, show_web=True,
                       size1={'width': 1500, 'height': 500, 'rotate': 15},
                       size2={'width': 1000, 'height': 500, 'rotate': 0}):
    '''
    用于展示train_data中全部数据集和分割后的训练集与验证集的各类数据分布
    用于展示train_data中图片尺寸数据分布
    :param data_path: train_data path
    :param show_web: 自动调用浏览器展示表格，默认值为True
    :param size1: {'width': 1500, 'height': 500, 'rotate': 15}
                    设置表格宽 默认值1500 单位px
                    设置表格高 默认值500 单位px
                    x轴数据倾斜角度 默认值15
    :param size2: {'width': 500, 'height': 500, 'rotate': 15}
                    设置表格宽 默认值500 单位px
                    设置表格高 默认值500 单位px
                    x轴数据倾斜角度 默认值0
    :return:
    '''
    train_json_path = os.path.join(data_path, 'train_img_dict.json')
    if not os.path.exists(train_json_path):
        raise FileNotFoundError('请先执行 data_shuffle_split 方法划分数据集')
    train_json_dict = __json_load(train_json_path)
    train_dict = dict(Counter(train_json_dict.values()))  # Counter统计元素出现的次数
    train_dict = dict(sorted(train_dict.items()))
    verify_json_path = os.path.join(data_path, 'verify_img_dict.json')
    if not os.path.exists(verify_json_path):
        raise FileNotFoundError('请先执行 data_shuffle_split 方法划分数据集')
    verify_json_dict = __json_load(verify_json_path)
    verify_dict = dict(Counter(verify_json_dict.values()))  # Counter统计元素出现的次数
    verify_dict = dict(sorted(verify_dict.items()))
    all_dict = dict([[k, train_dict[k] + verify_dict[k]] for k in train_dict])
    train_json_path_40 = os.path.join(data_path, 'train_img_dict_40.json')
    if not os.path.exists(train_json_path_40):
        raise FileNotFoundError('请先执行 data_shuffle_split 方法划分数据集')
    train_json_dict_40 = __json_load(train_json_path_40)
    train_dict_40 = dict(Counter(train_json_dict_40.values()))  # Counter统计元素出现的次数
    train_dict_40 = dict(sorted(train_dict_40.items()))
    verify_json_path_40 = os.path.join(data_path, 'verify_img_dict_40.json')
    if not os.path.exists(verify_json_path_40):
        raise FileNotFoundError('请先执行 data_shuffle_split 方法划分数据集')
    verify_json_dict_40 = __json_load(verify_json_path_40)
    verify_dict_40 = dict(Counter(verify_json_dict_40.values()))  # Counter统计元素出现的次数
    verify_dict_40 = dict(sorted(verify_dict_40.items()))
    all_dict_40 = dict([[k, train_dict_40[k] + verify_dict_40[k]] for k in train_dict_40])

    # 40分类数据可视化
    garbage_classify_rule = __json_load(
        os.path.join(data_path, 'garbage_classify_rule.json'))
    # x轴输数据
    x=["{}-{}".format(id, garbage_classify_rule[str(id)]) for id in train_dict_40.keys()]
    # y轴数据
    data_y = list(all_dict_40.values())
    train_y = list(train_dict_40.values())
    verify_y = list(verify_dict_40.values())
    # 创建Bar示例
    bar = Bar(init_opts=opts.InitOpts(width='{}px'.format(size1['width']), height='{}px'.format(size1['height'])))
    bar.add_xaxis(xaxis_data=x)
    bar.add_yaxis(series_name='All', yaxis_data=data_y)
    bar.add_yaxis(series_name='Train', yaxis_data=train_y)
    bar.add_yaxis(series_name='Verify', yaxis_data=verify_y)
    # 设置全局参数
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title='40分类\n垃圾分类 All/Train/Verify 不同类别数据分布'),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=size1['rotate']))  # 使x轴数据标签倾斜
    )
    # 展示图表
    bar.render('All_Train_Verify_40.html')
    bar_path = os.getcwd()
    if show_web:
        url = 'file://' + bar_path + '/All_Train_Verify_40.html'
        webbrowser.open(url)
    else:
        print('打开 \'' + bar_path + '/All_Train_Verify_40.html\'以查看图表')

    # 4分类数据可视化
    garbage_classify_rule = __json_load(
        os.path.join(data_path, 'garbage_classify_index.json'))
    # x轴输数据
    x = ["{}-{}".format(id, garbage_classify_rule[str(id)]) for id in train_dict.keys()]
    # y轴数据
    data_y = list(all_dict.values())
    train_y = list(train_dict.values())
    verify_y = list(verify_dict.values())
    # 创建Bar示例
    bar = Bar(init_opts=opts.InitOpts(width='{}px'.format(size1['width']), height='{}px'.format(size1['height'])))
    bar.add_xaxis(xaxis_data=x)
    bar.add_yaxis(series_name='All', yaxis_data=data_y)
    bar.add_yaxis(series_name='Train', yaxis_data=train_y)
    bar.add_yaxis(series_name='Verify', yaxis_data=verify_y)
    # 设置全局参数
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title='4分类\n垃圾分类 All/Train/Verify 不同类别数据分布'),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=size1['rotate']))  # 使x轴数据标签倾斜
    )
    # 展示图表
    bar.render('All_Train_Verify_4.html')
    bar_path = os.getcwd()
    if show_web:
        url = 'file://' + bar_path + '/All_Train_Verify_4.html'
        webbrowser.open(url)
    else:
        print('打开 \'' + bar_path + '/All_Train_Verify_4.html\'以查看图表')

# 图片尺寸数据直方图
def img_size_visualization(imgs_size):

    ratio_list = [item[3] for item in imgs_size]  # 获取img_size中的比例数据
    new_ratio_list = list(filter(lambda x: x > 0.5 and x <= 2, ratio_list))
    # 创建示例对象
    sns.set()
    np.random.seed(0)
    __set_zh()  # 设置中文
    # seaborn 直方图展示
    ax = sns.distplot(ratio_list)
    plt.title('原始数据分布')
    plt.show()
    ax = sns.distplot(new_ratio_list)  # 数据分布（0，2）
    plt.title('过滤后的数据分布（0.5<x<2）')
    plt.show()

# 进度展示
def __processBar(message, num, total):
    '''
    进度展示  message:  num/total   100%
    :param message: 消息
    :param num: 当前进度
    :param total: 总体
    :return:
    '''
    rate = num / total
    rate_num = int(rate * 100)
    if rate_num == 100:
        r = '\r{}:\t{}/{}\t100%\n'.format(message, num, total)
    else:
        r = '\r{}:\t{}/{}\t{}%'.format(message, num, total, rate_num)
    sys.stdout.write(r)
    sys.stdout.flush


# 解决matplotlib中文显示
def __set_zh():
    '''
    matplotlib中文显示
    :return:
    '''
    type = sys.platform
    if type == 'win32':
        plt.rcParams['font.sans-serif'] = ['KaiTi']
    elif type == 'linux':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        plt.rcParams['font.sans-serif'] = ['Songti SC']  # 正常显示中文标签

    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负数的负号显示问题


def __version__():
    device = ('GPU' if torch.cuda.is_available() else 'CPU')
    print('torch version:', torch.__version__)
    print('torchvision version:', torchvision.__version__)
    if torch.cuda.is_available():
        print('Device:GPU')
        print('GPU list:')
        for i in range(torch.cuda.device_count()):
            print('       {}. {}'.format(i, torch.cuda.get_device_name(i)))
        print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('Device:CPU')


# 测试
if __name__ == '__main__':
    __version__()
    Root_Path = ''
    Data_Path = Root_Path + 'data/'
    #img_path_list, img_name2label_dict, img_label2count_dict = get_img_info(Data_Path)
    #train_json_path, verify_json_path, train_img_list, verify_img_list = data_shuffle_split(0.8, img_path_list)
    data_visualization(Data_Path, show_web=False)
    imgs_size = get_img_size(Data_Path)
    img_size_visualization(imgs_size)