"""
Author: Yixuan Su
Date: 2025/04/03 15:01
File: obj_to_obj_rotmat.py
Description: 

"""
import numpy as np

# 定义绕 Z 轴顺时针旋转 90 度的旋转矩阵
Rz = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])

# 定义绕 Y 轴顺时针旋转 90 度的旋转矩阵
Ry = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])

# 组合旋转矩阵
R = np.dot(Ry, Rz)

# 输出旋转矩阵
print("最终旋转矩阵 R:")
print(R)
