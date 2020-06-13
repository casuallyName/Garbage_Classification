# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 23:25
# @Author: Casually
# @File: forecast.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
# from _collections import OrderedDict
import torch
from flask import Flask, request, jsonify
from utils.json_utils import jsonify
from utils.model import init_model
from utils.transform import transform_image
import time,os
from collections import OrderedDict
from args import args

# 实例化
app = Flask(__name__)


# 设置编码-否则返回数据中文时候-乱码
app.config['JSON_AS_ASCII'] = False

# 加载Label2Name Mapping
class_id2name=args.class_id2name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
print('{} 设备类型：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), str(device).upper()))
num_classes = len(class_id2name)
model_name = 'ResNeXt_32x16d_WSL_Attention'
model_ft = init_model(model_name, num_classes, feature_extract=True)
model_ft.to(device)  # 设置模型运行环境
model_path = './pth_files/ResNeXt_32x16d_WSL_Attention.pth'

print('{} 模型路径：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), model_path))
# 显存不够了，放内存上吧
model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))

model_ft.eval()


# 加载模型，加载模型完显存不够了，只能放到CPU上了...

# route()方法用于设定路由；类似spring路由配置
@app.route('/')  # 测试
def test():
    result = OrderedDict(error=0, errmsg='success', data=None)
    return jsonify(result)


@app.route('/forecast', methods=['POST'])  # 在线预测
def forecast():
    # 获取输入数据
    file = request.files['image']
    img_bytes = file.read()

    feature = transform_image(img_bytes) # 特征提取
    feature = feature.to(device)  # 在device 上进行预测

    # 模型预测
    with torch.no_grad():
        t1 = time.time()
        outputs = model_ft.forward(feature)
        consume = (time.time() - t1)
        consume = round(consume, 3)

    # API 结果封装
    label_c_mapping = {}
    # 通过softmax 获取每个label的概率
    outputs = torch.nn.functional.softmax(outputs[0], dim=0)
    pred_list = outputs.cpu().numpy().tolist()

    for i, prob in enumerate(pred_list):
        label_c_mapping[int(i)] = prob

    ## 按照prob 降序，获取topK = 5
    dict_list = []
    for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:5]:
        label = int(label_prob[0])
        result = {'label': label, 'acc': label_prob[1], 'name': class_id2name[label]}
        dict_list.append(result)

    ## dict 中的数值按照顺序返回结果
    result = OrderedDict(error=0,
                         errmsg='success',
                         consume=consume,
                         label=dict_list[0]['label'],
                         acc='{}%'.format(int(dict_list[0]['acc'] * 100)),
                         name=dict_list[0]['name'],
                         data=dict_list)
    return jsonify(result)


if __name__ == '__main__':
    app.run(
        host='192.168.3.9'
    )
