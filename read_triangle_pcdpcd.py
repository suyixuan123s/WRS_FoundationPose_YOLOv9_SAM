"""
Author: Yixuan Su
Date: 2024/11/17 10:26
File: read_triangle_mesh.py
"""

import os
import open3d as o3d

pcd = o3d.io.read_point_cloud("rE:\ABB-Project\ABB_wrs_huri_su\huri\models\rack.stl")

# 检查是否加载成功
if pcd.is_empty():
    print("加载 PCD 文件失败,请检查路径或文件格式")
else:
    # 为 mesh 添加颜色
    pcd.paint_uniform_color([0, 0.5, 0])  # RGB 格式

o3d.visualization.draw_geometries([pcd])

#
# # 加载 STL 文件
#
#
# file_path = r"E:\ABB-Project\ABB_wrs_huri_su\huri\models\rack.stl"
# mesh = o3d.io.read_point_cloud(file_path)
#
# # 检查是否加载成功
# if mesh.is_empty():
#     print("加载 PCD 文件失败,请检查路径或文件格式")
# else:
#     # 为 mesh 添加颜色
#     mesh.paint_uniform_color([0, 0.5, 0])  # RGB 格式
#     # 创建坐标系
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # 坐标系大小为0.1
#
#     # 显示 mesh 和坐标系
#     o3d.visualization.draw_geometries([mesh, coordinate_frame], window_name="STL 文件显示 - 带颜色和原点",
#                                       width=1280, height=720, mesh_show_wireframe=False, mesh_show_back_face=False)
#
#     # # 显示 mesh
#     # o3d.visualization.draw_geometries([mesh], window_name="STL 文件显示 - 带颜色",
#     #                                   width=1280, height=720, mesh_show_wireframe=False, mesh_show_back_face=False)
