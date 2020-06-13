# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:43
# @Author: Casually
# @File: ProgressBar.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
# 进度条
import sys, time


def show(batch, size, msg_behind='', msg_front='', time_show=True):
    '''
    进度条展示
    :param batch:
    :param size:
    :param msg_behind:
    :param msg_front:
    :param time_show:
    :return:
    '''
    if time_show:
        if (batch + 1) < size:
            sys.stdout.write('\r{} {}[{}/{}]{}'.format(
                time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
                msg_behind,
                batch + 1,
                size,
                msg_front
            ))
            sys.stdout.flush()
        elif (batch + 1) == size:
            sys.stdout.write('\r{} {}[{}/{}]{}\n'.format(
                time.strftime("[%d/%b/%Y %H:%M:%S]", time.localtime()),
                msg_behind,
                batch + 1,
                size,
                msg_front
            ))
        else:
            raise ValueError(' batch > size')
    else:
        if (batch + 1) < size:
            sys.stdout.write('\r{}[{}/{}]{}'.format(
                msg_behind,
                batch + 1,
                size,
                msg_front
            ))
            sys.stdout.flush()
        elif (batch + 1) == size:
            sys.stdout.write('\r{}[{}/{}]{}\n'.format(
                msg_behind,
                batch + 1,
                size,
                msg_front
            ))
        else:
            raise ValueError(' batch > size')
