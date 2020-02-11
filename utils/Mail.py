# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/2/10 13:16
# @Author: Casually
# @File: Mail.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr




def mail(to_user,subject='',info=''):
    my_sender = '841765793@qq.com'  # 发件人邮箱账号
    my_pass = 'woqurjpnfnxrbfdc'  # 发件人邮箱密码

    try:
        msg = MIMEText(info, 'plain', 'utf-8')
        msg['From'] = formataddr(['训练状态', my_sender])  # 发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(['', to_user])  # 收件人邮箱昵称、收件人邮箱账号
        msg['Subject'] = subject # 邮件主题

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, [to_user, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
        return True
    except Exception:
        return False



if __name__=='__main__':
    ret = mail(
               to_user= 'fjklqq@163.com',
                subject='TEST',
               info='Test\ndemo'
               )
    if ret:
        print("邮件发送成功")
    else:
        print("邮件发送失败")