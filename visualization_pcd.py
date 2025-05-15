"""
Author: Yixuan Su
Date: 2025/03/10 17:13
File: visualization_pcd.py
Description: 

"""
import open3d as o3d
pcd = o3d.io.read_triangle_mesh(r"E:\ABB-Project\ABB_wrs_hu\suyixuan\data\out_rack\textured_mesh.obj")
o3d.visualization.draw_geometries([pcd])