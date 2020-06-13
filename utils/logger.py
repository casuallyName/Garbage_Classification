# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:18
# @Author: Casually
# @File: logger.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
#  日志文件

__all__ = ['Logger']


class Logger(object):
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:  # 判断重新开辟日志还是续写日志
                self.file = open(fpath, 'r')  # 只读打开
                name = self.file.readline()
                self.names = name.rstrip().split(' | ')  # 从现有日志中读取标题
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []  # 为标题创建字典

                for numbers in self.file:
                    numbers = numbers.rstrip().split(' | ')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])  # 从现有日志读取数据
                self.file.close()  # 关闭文件
                self.file = open(fpath, 'a')  # 追加写打开文件
            else:
                self.file = open(fpath, 'w')  # 打开一个新的文件

    def set_names(self, names):
        '''
        Log头信息
        :param names:
        :return:
        '''
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write(' | ')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()  # 方法是用来刷新缓冲区的，将缓冲区中的数据立刻写入文件，同时清空缓冲区

    def append(self, numbers):
        '''
        追加数据
        :param numbers:
        :return:
        '''
        # 判断数据个数是否与标题个数相等
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        # 开始遍历数据写入
        for index, num in enumerate(numbers):
            if type(num) == float:
                self.file.write("{:.4f}".format(num))
            else:
                self.file.write("{}".format(num))
            self.file.write(' | ')
            self.numbers[self.names[index]].append(num)  # 将新数据也添加到成员变量中
        self.file.write('\n')
        self.file.flush()  # 方法是用来刷新缓冲区的，将缓冲区中的数据立刻写入文件，同时清空缓冲区

    def close(self):
        '''
        关闭文件
        :return:
        '''
        if self.file is not None:
            self.file.close()
