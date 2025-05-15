"""
Author: Yixuan Su
Date: 2025/04/03 15:12
File: compute_difference.py
Description: 

"""
import numpy as np

# 定义两个位置的坐标(例如,P1 和 P2)
P1 = np.array([.647, -.242, .0475])
P2 = np.array([0.64498483, -0.26111259, 0.06005253])

# 计算差值
difference = P2 - P1

# 输出差值
print("位置差值:", difference)
