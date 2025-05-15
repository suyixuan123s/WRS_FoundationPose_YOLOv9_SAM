"""
# Time： 2025/2/18 17:40
# Author： Yixuan Su
# File： Visualization_Mesh.py
# IDE： PyCharm
# Motto：ABC(Never give up!)
# Description：
"""


import os
import open3d as o3d
import numpy as np

pcd = o3d.t.io.read_triangle_mesh("/debug_mustard0/model_tf.obj")

o3d.visualization.draw([pcd])
