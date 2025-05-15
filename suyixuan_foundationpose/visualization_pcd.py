#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FoundationPose 
@File    ：visualization_mesh.py
@IDE     ：PyCharm 
@Author  ：suyixuan44444444444444444444444444
@Date    ：2025-02-24 14:31:04 
'''


import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/suyixuan/AI/Pose_Estimation/sam2/data/hujiaying/424/shiguanjia/ply/color_image_20250424-153409.ply")

o3d.visualization.draw_geometries([pcd])