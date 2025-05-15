"""
Author: Yixuan Su
Date: 2025/03/03 04:12
File: valisulazation_pcd.py
Description: 

"""
import open3d as o3d

pcd = o3d.io.read_triangle_mesh(r"E:\ABB-Project\ABB_wrs\0000_placementplanner\objects\test_long_small.stl")

o3d.visualization.draw_geometries([pcd])




