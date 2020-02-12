# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 23:25
# @Author: Casually
# @File: server.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
from _collections import OrderedDict
import torch
# 导入Flask类
from flask import Flask, request, jsonify
from utils.json_utils import jsonify
from utils.model import initital_model
from utils.transform import transform_image
import time,os
from collections import OrderedDict
import codecs
from args import args


# 实例化
app = Flask(__name__)

# 获取所有配置参数
state = {k: v for k, v in args._get_kwargs()}

# 设置编码-否则返回数据中文时候-乱码
app.config['JSON_AS_ASCII'] = False

# 加载Label2Name Mapping
class_id2name=args.class_id2name
# class_id2name = {}
# for line in codecs.open('utils/garbage_label.txt', 'r', encoding='utf-8'):
#     line = line.strip()
#     _id = line.split(":")[0]
#     _name = line.split(":")[1]
#     class_id2name[int(_id)] = _name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
print('{} 设备类型：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), str(device).upper()))
num_classes = len(class_id2name)
model_name = args.model_name
model_ft = initital_model(model_name, num_classes, feature_extract=True)
model_ft.to(device)  # 设置模型运行环境
model_path = os.path.join(args.pth_path,args.pth_name)
#'pth_files/garbage_resnext101_model_7_9417_9500.pth'
print('{} 模型路径：\t{}'.format(time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()), model_path))
# 显存不够了，放内存上吧
model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
# note::
# When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
#         will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
#         and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.
model_ft.eval()


# 加载模型，加载模型完显存不够了，只能放到CPU上了...

# route()方法用于设定路由；类似spring路由配置
@app.route('/')  # 测试
def test():
    result = OrderedDict(error=0, errmsg='success', data='惊不惊喜！')
    # print('测试\t',end = '')
    return jsonify(result)


@app.route('/forecast', methods=['POST'])  # 在线预测
def forecast():
    # 获取输入数据
    file = request.files['img']
    img_bytes = file.read()

    # 特征提取
    feature = transform_image(img_bytes)
    feature = feature.to(device)  # 在device 上进行预测

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
    # app.run(host, port, debug, options)
    # 默认值：host=127.0.0.1, port=5000, debug=false
    app.run(
        # host='172.33.23.55'  # 使用VPN组网时IP段
        host='192.168.1.100'  # 使用内网穿透时的URL http://http://eason.iask.in/
    )
