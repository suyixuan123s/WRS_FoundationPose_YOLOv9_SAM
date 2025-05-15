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


# 定义 4x4 矩阵
# object_to_camera = np.array([
#     [0.9986197948455810547, -0.01462547201663255692, -0.05044681578874588013, 0.1561549007892608643],
#     [0.01522155944257974625, 0.9998186230659484863, 0.01145211141556501389, 0.1472180634737014771],
#     [0.05027017742395401001, -0.01220416184514760971, 0.9986611008644100400, 0.8086225390434265137],
#     [0.0000000000000000000, 0.0000000000000000000, 0.0000000000000000000, 1.0000000000000000000]
# ])

object_to_camera = np.array(
    [[0.9943566918373109, 0.052462488412857056, -0.09220805019140243, 0.16108790040016174],
     [-0.026044327765703201, -0.7218495011329651, -0.6915596723556519, 0.1449363082647324],
     [-0.10284128040075302, 0.6900585293769836, -0.7164096236228943, 0.8313495516777039],
     [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

# object_to_object1 = np.array([])


# object_to_camera = np.array(
#     [[0.9995586276054382324, 0.0004511209263000637293, -0.02969847433269023895, 0.1589225083589553833],
#      [-0.0001064394673448987305, 0.9999325871467590332, 0.01160741318017244339, 0.1440123915672302246],
#      [0.02970174513757228851, -0.01159908436238765717, 0.9994915723800659180, 0.7972096204757690430],
#      [0.000000, 0.000000, 0.000000, 1.000000]])

# 计算物体到世界坐标系的变换矩阵
transformation_object_to_world = transformation_camera_to_sim @ object_to_camera

# 指定文件路径
file_path = r"E:\ABB-Project\ABB_wrs\suyixuan\Transformation_Object_to_World\transformation_object_to_world.txt"

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