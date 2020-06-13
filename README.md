# 垃圾分类 Garbage_Classification(G_C)

## 1. 函数说明

* main.py 
* 模型训练主程序
* args.py 

  * 保存模型参数
* data_preprocess.py

  * 数据预处理相关
* forecast.py

  * 模型预测使用，Web接口方式
  * 结构格式：
    * 请求地址：{Host}:{Point}/forecast
    * 请求方式：Post
    * Body：
      * KEY: image
      * VALUE: 图像文件
* models（Folder）

  * 存放各模型
* utils（Folder）

  * 存放其他相关功能函数

## 2.测试版本

| 类型    | 版本 |
| ------- | ---- |
| Python  | 3.7  |
| PyTorch | 1.5  |
| CUDA    | 10.2 |

