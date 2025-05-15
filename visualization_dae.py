"""
Author: Yixuan Su
Date: 2025/03/10 21:56
File: visualization_dae.py
Description: 

 Uninstalling PyOpenGL-3.1.9
 Successfully installed PyOpenGL-3.1.0 freetype-py-2.5.1 pyassimp-5.2.5 pyrender-0.1.45

(wrs_wan) C:\Users\21344>conda list pyglet
# packages in environment at D:\Anaconda3\envs\wrs_wan:
#
# Name                    Version                   Build  Channel
pyglet                    2.1.3                    pypi_0    pypi

(wrs_wan) C:\Users\21344>

"""



import trimesh

# 使用原始字符串路径
path = "/wrs/robot_sim/manipulators/gofa10/meshes/base.dae"

# 加载并显示网格
mesh = trimesh.load_mesh(path)

# 显示网格
mesh.show()

