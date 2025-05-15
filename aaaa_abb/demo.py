"""
Author: Yixuan Su
Date: 2025/02/20 19:52
File: demo.py
Description: 

"""
from direct.task.TaskManagerGlobal import taskMgr

import visualization.panda.world as wd
import humath as hm
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import modeling.geometric_model as gm
import os
import trimesh as trimesh
import trimeshwraper as tw
import vision.depth_camera.pcd_data_adapter as vdda
import open3d as o3d

# 确保使用正确的缩放因子
name = "rack_5ml_green.STL"
obj = tw.TrimeshHu(r"E:\ABB-Project\ABB_wrs\suyixuan\ABB\Pose_Estimation\Task5_ICP_GOFA5\mesh", name, scale=1.0)  # 或者根据需要调整

# 打印模型的顶点信息以确认加载
print("Vertices:", obj.objtrm.vertices)

# 进行采样
sample = obj.objtrm.vertices
for pnt in sample:
    gm.gen_sphere(pnt, 0.3, [0, 1, 0, 1]).attach_to(gm.base)

# 确保模型附加到场景
obj.attach_to(gm.base)

# 运行渲染
gm.base.run()