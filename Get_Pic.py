# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/2/6 22:28
# @Author: Casually
# @File: Get_Pic.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
import itertools
import urllib
import requests
import os, re, sys
import time

keyword_list = [
    [0, 0, '一次性快餐盒', 500],
    [1, 0, '塑料', 400],
    [2, 0, '烟蒂', 500],
    [3, 0, '牙签', 600],
    [4, 0, '破碎花盆及碟碗', 400],
    [5, 0, '竹筷', 400],
    [6, 1, '剩饭剩菜', 300],
    [7, 1, '大骨头', 400],
    [8, 1, '果皮', 400],
    [9, 1, '水果', 400],
    [10, 1, '茶叶渣', 400],
    [11, 1, '菜叶菜根', 10],
    [12, 1, '蛋壳', 400],
    [13, 1, '鱼刺鱼骨', 300],
    [14, 2, '充电宝', 400],
    [15, 2, '包', 300],
    [16, 2, '化妆品瓶子', 400],
    [17, 2, '塑料玩具', 300],
    [18, 2, '塑料碗盆', 400],
    [19, 2, '塑料衣架', 300],
    [20, 2, '快递纸袋', 500],
    [21, 2, '插头电线', 100],
    [22, 2, '旧衣服', 400],
    [23, 2, '易拉罐', 400],
    [24, 2, '枕头', 400],
    [25, 2, '毛绒玩具', 400],
    [26, 2, '洗发水瓶', 400],
    [27, 2, '玻璃杯', 200],
    [28, 2, '皮鞋', 400],
    [29, 2, '砧板', 300],
    [30, 2, '纸板箱', 400],
    [31, 2, '调料瓶', 300],
    [32, 2, '酒瓶', 500],
    [33, 2, '金属食品罐', 400],
    [34, 2, '锅', 400],
    [35, 2, '食用油桶', 400],
    [36, 2, '饮料瓶', 500],
    [37, 3, '干电池', 400],
    [38, 3, '软膏', 350],
    [39, 3, '药盒', 300],
    [40, 3, '药粒', 100]
]

str_table = {
    '_z2C$q': ':',
    '_z&e3B': '.',
    'AzdH3F': '/'
}

char_table = {
    'w': 'a',
    'k': 'b',
    'v': 'c',
    '1': 'd',
    'j': 'e',
    'u': 'f',
    '2': 'g',
    'i': 'h',
    't': 'i',
    '3': 'j',
    'h': 'k',
    's': 'l',
    '4': 'm',
    'g': 'n',
    '5': 'o',
    'r': 'p',
    'q': 'q',
    '6': 'r',
    'f': 's',
    'p': 't',
    '7': 'u',
    'e': 'v',
    'o': 'w',
    '8': '1',
    'd': '2',
    'n': '3',
    '9': '4',
    'c': '5',
    'm': '6',
    '0': '7',
    'b': '8',
    'l': '9',
    'a': '0'
}

# str 的translate方法需要用单个字符的十进制unicode编码作为key
# value 中的数字会被当成十进制unicode编码转换成字符
# 也可以直接用字符串作为value
char_table = {ord(key): ord(value) for key, value in char_table.items()}


# 解码图片URL
def decode(url):
    # 先替换字符串
    for key, value in str_table.items():
        url = url.replace(key, value)
    # 再替换剩下的字符
    return url.translate(char_table)


# 生成网址列表
def buildUrls(word):
    word = urllib.parse.quote(word)
    url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
    urls = (url.format(word=word, pn=x) for x in itertools.count(start=0, step=60))
    return urls


# 解析JSON获取图片URL
re_url = re.compile(r'"objURL":"(.*?)"')


def resolveImgUrl(html):
    imgUrls = [decode(x) for x in re_url.findall(html)]
    return imgUrls


def downImg(imgUrl, dirpath, imgName):
    filename = os.path.join(dirpath, imgName)
    try:
        res = requests.get(imgUrl, timeout=15)
        if str(res.status_code)[0] == "4":
            print(str(res.status_code), ":", imgUrl)
            return False
    except Exception as e:
        print('{Time} 抛出异常：\t{Error}'.format(
            Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
            Error=e))
        return False
    with open(filename, "wb") as f:
        f.write(res.content)
    return True


def mkDir(dirName):
    dirpath = os.path.join(sys.path[0], dirName)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


if __name__ == '__main__':
    for item in keyword_list:
        word = item[2]
        imageNum = item[3];  # 下载图片的数目
        saveImagePath = "New_Pic/{}/{}/".format(item[1], item[0])  # 保存图片的途径
        print('{Time} 爬取内容：\t{data}'.format(
            Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
            data='{} * {}'.format(word, imageNum)
        ))
        print('{Time} 存储位置：\t{Path}'.format(
            Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
            Path=saveImagePath
        ))
        indexOffset = 0  # 图像命名起始点
        dirpath = mkDir(saveImagePath)
        urls = buildUrls(word)
        index = 0
        flag = 0
        for url in urls:
            print('{Time} 正在请求网络：'.format(
                Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime())), end='')
            html = requests.get(url, timeout=10).content.decode('utf-8')
            imgUrls = resolveImgUrl(html)
            if len(imgUrls) == 0:  # 没有图片则结束
                print('失败')
                break
            else:
                print('完成')
            for url in imgUrls:
                img_name = "img_{}_{}_{}.jpg".format(item[1], item[0], index + indexOffset)
                if downImg(url, dirpath, img_name):
                    index += 1
                    flag = 0
                    print("保存  {:>4}".format(img_name))
                if index == imageNum:
                    break
            if flag > 100:
                break
            if index == imageNum:
                print('{Time} 下载完成，存储路径：\t{Path}%'.format(
                    Time=time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
                    Path=saveImagePath))
                break
