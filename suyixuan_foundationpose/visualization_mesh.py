#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：FoundationPose 
@File    ：visualization_mesh.py
@IDE     ：PyCharm 
@Author  ：suyixuan_sam
@Date    ：2025-02-24 14:31:04 
'''


import numpy as np
import open3d as o3d

pcd = o3d.io.read_triangle_mesh("/home/suyixuan/AI/Pose_Estimation/BundleSDF/data/qxx/single_tack_w_out/textured_mesh.obj")

o3d.visualization.draw_geometries([pcd])