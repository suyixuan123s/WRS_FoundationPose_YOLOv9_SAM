"""
Created on 4/29/2024 
Author: Hao Chen (chen960216@gmail.com)
"""
import numpy as np


def Mm(n):
    return n / 1000


# TBM_ARM
TBM_ARM_ORIGIN2BASE = Mm(485) - Mm(30)  # 30 mm is baffle

# TABLE
# ORIGIN2TABLE = np.array([Mm(-30), -Mm(200), -1.255])
ORIGIN2TABLE = np.array([Mm(-30), -Mm(200), -0.86705263])
ORIGIN2CONVEYOR_TABLE = np.array([0, 0, 0])

# cutter conveyor
# ORIGIN2TBM_CUTTERCONVEYOR_ORIGIN = np.array([-Mm(158), -Mm(833), -1.255])
ORIGIN2TBM_CUTTERCONVEYOR_ORIGIN = np.array([-Mm(113), -Mm(845.5), -Mm(878.33466)])
# AUTO GATE # np.array([3.99, -0.001, 0.287])
ORIGIN2TBM_AUTOGATE_ORIGIN = np.array([Mm(3960), 0, Mm(363)])
AUTOGATE_OPEN_RANGE = [0, np.radians(102)]

# tbm cutters
ORIGIN2TBM_TBMCUTTERS_ORIGIN = np.array([Mm(5900), Mm(881.000), -Mm(2922)])
