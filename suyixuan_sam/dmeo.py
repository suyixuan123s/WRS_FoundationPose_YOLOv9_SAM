"""
Author: Yixuan Su
Date: 2025/04/01 23:01
File: dmeo.py
Description: 

"""
import cv2
import numpy as np

# 读取图像
img_path = r'E:\ABB\segment-anything\suyixuan\Canny\demo.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)



# 对图像进行二值化处理
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 使用轮廓检测来查找烟盒的轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤掉小的噪声轮廓（假设烟盒的面积较大）
min_area = 1000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 绘制轮廓并计数
img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 255, 0), 2)

# 显示轮廓并返回结果
num_boxes = len(filtered_contours)
img_with_contours_show = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)

num_boxes, img_with_contours_show
