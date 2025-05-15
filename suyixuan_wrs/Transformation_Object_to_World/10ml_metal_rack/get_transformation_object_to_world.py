"""
Author: Yixuan Su
Date: 2025/03/29 20:26
File: get_transformation_object_to_world.py
Description: 

"""
import os
import open3d as o3d
import numpy as np

transformation_camera_to_sim = np.array([[0.995588, - 0.035402, - 0.086900, 0.565000],
                                         [0.034767, - 0.721015, 0.692047, - 0.720000],
                                         [-0.087156, - 0.692015, - 0.716602, 0.755000],
                                         [0.000000, 0.000000, 0.000000, 1.000000]])

object_to_camera = np.array([
    [-0.003554405644536018, 0.9985489845275879, 0.05373386666178703, 0.3613424301147461],
    [-0.7364977002143859, -0.03896169364452362, 0.6753169894218445, -0.03765789046883583],
    [0.6764307022094727, -0.037174560129642486, 0.735567569732666, 0.9950773715972901],
    [0.0, 0.0, 0.0, 1.0]
])

# 计算物体到世界坐标系的变换矩阵
transformation_object_to_world = transformation_camera_to_sim @ object_to_camera

# 指定文件路径
file_path = r"E:\ABB-Project\ABB_wrs\suyixuan\Transformation_Object_to_World\10ml_metal_rack\transformation_object_to_world.txt"

# 检查目录是否存在,如果不存在则创建
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 将结果保存到文件
with open(file_path, "w", encoding="utf-8") as f:
    f.write("相机到仿真坐标系的外参 (Camera to Simulation):\n")
    f.write(np.array2string(transformation_object_to_world, formatter={'float_kind': lambda x: f"{x:.6f}"}))
print(f"文件已保存到: {file_path}")

print("Object to World Transformation Matrix:")
print(transformation_object_to_world)

# # T = rm.homomat_from_posrot(pos=np.array([0.37, -0.68, 0.53]),  # TODO
# #                            rot=rm.rotmat_from_euler(ai=np.pi * (-131) / 180, aj=np.pi * (-1.5) / 180,
# #                                                     ak=np.pi * (-3) / 180))
# K = [[606.6265258789062, 0, 324.2806701660156], [0, 606.6566772460938, 241.14862060546875], [0, 0, 1]]
#
# # 转换到像素坐标系,找到像素坐标系还是图像坐标系.   点是像素坐标系. 图像坐标系是中心点.  内参是什么转化到哪里？？
# inv_T = np.linalg.inv(camera_to_table_transform)
# obj_camera = np.dot(inv_T, obj_tablepos.T)
#
# obj_camera_noralmized=obj_camera[:3]/obj_camera[2]
# obj_pic = K@ obj_camera_noralmized
# print("obj_ti   in pic pos :", obj_pic)


# Object to World Transformation Matrix:
# [[ 0.99257126 -0.03394252 -0.11683419  0.6488455 ]
#  [ 0.0553834  -0.72897782  0.6822935  -0.26660331]
#  [-0.1083282  -0.68369574 -0.72168176  0.07020821]
#  [ 0.          0.          0.          1.        ]]
