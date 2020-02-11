# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/2/7 20:14
# @Author: Casually
# @File: File_change.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
# -*- coding: utf-8 -*-

import os
import shutil

from utils.transform import transform_image
from utils import ProgressBar as Bar
import torch
# 导入Flask类
from utils.model import initital_model
import time
import codecs
from args import args


def __init__():
    # 获取所有配置参数
    state = {k: v for k, v in args._get_kwargs()}
    # 加载Label2Name Mapping
    class_id2name = {}
    for line in codecs.open('utils/garbage_label.txt', 'r', encoding='utf-8'):
        line = line.strip()
        _id = line.split(":")[0]
        _name = line.split(":")[1]
        class_id2name[int(_id)] = _name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    print('{} 设备类型：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), str(device).upper()))
    num_classes = len(class_id2name)
    model_name = args.model_name
    model_path = 'pth_files/2020_02_05_01_30_58.pth'
    print('{} 模型路径：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), model_path))

    model_ft = initital_model(model_name, num_classes, feature_extract=True)
    model_ft.to(device)  # 设置模型运行环境
    # 显存不够了，放内存上吧
    model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
    # note::
    # When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
    #         will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
    #         and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.
    model_ft.eval()
    return model_ft


def getFileName(file_dir):
    Files = []
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        Files += files
        # print(files)  # 当前路径下所有非目录子文件
    return Files


def fileCopy(files, from_path, to_path):
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    for i, file in enumerate(files):
        Bar.show(
            msg_behind='拷贝文件：',
            batch=i,
            size=len(files)
        )
        shutil.copy('{}/{}/{}/{}'.format(from_path, file.split('_')[1], file.split('_')[2], file),
                    '{}/{}'.format(to_path, file))


def fileNameChangeAndMark(files, path, start_Num=0):
    for i, file in enumerate(files):
        Bar.show(
            msg_behind='整理文件：',
            batch=i,
            size=len(files)
        )
        os.rename('{}/{}'.format(path, file), '{}/{}'.format(path, 'img_{}.jpg'.format(start_Num)))
        with open('{}/img_{}.txt'.format(path, start_Num), 'w') as f:
            f.write('img_{}.jpg, {}'.format(start_Num, file.split('_')[1]))
        start_Num += 1


def picFilter(files, path, to_path, model_ft, start_Num=0):
    for file in files:
        with open('{}/{}/{}/{}'.format(path, file.split('_')[1], file.split('_')[2], file), 'rb') as f:
            # 获取输入数据
            img_bytes = f.read()

            # 特征提取
            try:
                feature = transform_image(img_bytes)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                feature = feature.to(device)  # 在device 上进行预测
            except:
                continue

            # 模型预测
            with torch.no_grad():
                t1 = time.time()
                outputs = model_ft.forward(feature)
                consume = (time.time() - t1)  # * 1000
                consume = round(consume, 3)

            # API 结果封装
            label_c_mapping = {}
            ## The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            ## 通过softmax 获取每个label的概率
            outputs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_list = outputs.cpu().numpy().tolist()

            for i, prob in enumerate(pred_list):
                label_c_mapping[int(i)] = prob

            ## 按照prob 降序，获取topK = 5
            dict_list = []
            for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:5]:
                label = int(label_prob[0])
                dict_list.append([file, label, label_prob[1] * 100])
                # result = {'label': label, 'acc': label_prob[1], 'name': class_id2name[label]}
                # dict_list.append(result)
            # print(dict_list[0])
            if int(dict_list[0][0].split('_')[1]) == dict_list[0][1] and (95 < dict_list[0][2] < 100):
                print('拷贝：'+file)
                shutil.copy('{}/{}/{}/{}'.format(path, file.split('_')[1], file.split('_')[2], file),
                            '{}/{}'.format(to_path, file))
                os.rename('{}/{}'.format(to_path, file), '{}/{}'.format(to_path, 'img_{}.jpg'.format(start_Num)))
                with open('{}/img_{}.txt'.format(to_path, start_Num), 'w') as f:
                    f.write('img_{}.jpg, {}'.format(start_Num, file.split('_')[1]))
                start_Num += 1

            ## dict 中的数值按照顺序返回结果
            # result = dict(error=0,
            #                      errmsg='success',
            #                      consume=consume,
            #                      label=dict_list[0]['label'],
            #                      acc='{}%'.format(int(dict_list[0]['acc'] * 100)),
            #                      name=dict_list[0]['name'],
            #                      data=dict_list)
            # print(result)


if __name__ == '__main__':
    model_ft = __init__()
    path = 'New_Pic'
    to_path = 'data/All_data'
    files = getFileName(path)
    # fileCopy(files, path, to_path)
    picFilter(files, path, to_path, model_ft, start_Num=19736)
    # fileNameChangeAndMark(files, to_path, 19736)
