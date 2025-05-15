"""
Author: Yixuan Su
Date: 2025/04/03 15:33
File: rotation_matrix_axis.py
Description:
"""

import numpy as np


def rotation_matrix_x(theta):
    """绕X轴旋转的旋转矩阵"""
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])


def rotation_matrix_y(theta):
    """绕Y轴旋转的旋转矩阵"""
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rotation_matrix_z(theta):
    """绕Z轴旋转的旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def rotate_point(point, angles):
    """旋转一个点,angles是一个包含绕X、Y、Z轴旋转角度的元组"""
    x_angle, y_angle, z_angle = angles

    # 将角度转换为弧度
    x_angle = np.radians(x_angle)
    y_angle = np.radians(y_angle)
    z_angle = np.radians(z_angle)

    # 计算旋转矩阵
    R_x = rotation_matrix_x(x_angle)
    R_y = rotation_matrix_y(y_angle)
    R_z = rotation_matrix_z(z_angle)

    # 组合旋转矩阵(先绕Z轴旋转,再绕Y轴旋转,最后绕X轴旋转)
    R = R_z @ R_y @ R_x

    # 旋转点
    rotated_point = R @ point
    return rotated_point, R_x, R_y, R_z, R


# 示例
point = np.array([1, 0, 0])  # 要旋转的点
angles = (90, 0, 0)  # 绕X轴旋转90度

rotated_point = rotate_point(point, angles)
print("原始点:", point)
print("旋转后的点:", rotated_point)
