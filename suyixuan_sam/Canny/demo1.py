"""
Author: Yixuan Su
Date: 2025/04/01 23:02
File: demo1.py
Description: 

"""
import cv2

# 读取图像
image_path = r'E:\ABB\segment-anything\suyixuan\Canny\demo.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查路径是否正确。")
else:
    # 获取图像的大小
    height, width, channels = image.shape

    # 输出图像的大小
    print(f"图像的宽度: {width} 像素")
    print(f"图像的高度: {height} 像素")
    print(f"图像的通道数: {channels}")  # 通常为3（RGB）或1（灰度图）


import cv2
import numpy as np
image = cv2.imread(image_path)
