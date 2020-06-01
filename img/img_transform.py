#!/usr/bin/env python       增强代码可移植性
#-*- coding:utf-8 -*-       统一字符集
# @Time : 2020/5/7 下午10:04   创建脚本时间
# @Author :  KITATINE            申明作者
# @File   :  img_transform.py     申明文件名称
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 注意，只有PIL读取的图片才能被transforms接受

img = Image.open("./output.png")

resize = transforms.Resize([224, 224])
img = resize(img)
img.save('./01_tuning_patch/test_output.png')



