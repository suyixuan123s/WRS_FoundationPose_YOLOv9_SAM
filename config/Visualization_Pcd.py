"""
# Time： 2025/2/20 10:23
# Author： Yixuan Su
# File： Visualization_Pcd.py
# IDE： PyCharm
# Motto：ABC(Never give up!)
# Description：
"""

import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/suyixuan/AI/Pose_Estimation/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj")
o3d.visualization.draw_geometries([pcd])
