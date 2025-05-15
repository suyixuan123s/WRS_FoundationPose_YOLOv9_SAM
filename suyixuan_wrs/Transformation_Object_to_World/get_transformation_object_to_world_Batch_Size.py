"""
Author: Yixuan Su
Date: 2025/03/29 20:26
File: get_transformation_object_to_world.py
Description: 

"""
import os
import numpy as np

# 相机到仿真坐标系的变换矩阵
transformation_camera_to_sim = np.array([[0.995588, -0.035402, -0.086900, 0.565000],
                                         [0.034767, -0.721015, 0.692047, -0.720000],
                                         [-0.087156, -0.692015, -0.716602, 0.755000],
                                         [0.000000, 0.000000, 0.000000, 1.000000]])

# 指定包含 object_to_camera 矩阵的文件夹路径
input_folder = r"E:\ABB-Project\ABB_wrs\suyixuan\Transformation_Object_to_World\Fob_in_cam"
output_folder = r"E:\ABB-Project\ABB_wrs\suyixuan\Transformation_Object_to_World\tran_object_world"

# 检查输出目录是否存在,如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .txt 文件并排序
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
# 自定义排序函数,按文件名中的数字排序
txt_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) or 0)

# 遍历排序后的文件
for filename in txt_files:
    file_path = os.path.join(input_folder, filename)

    # 读取 object_to_camera 矩阵
    object_to_camera = np.loadtxt(file_path)

    # 计算物体到世界坐标系的变换矩阵
    transformation_object_to_world = transformation_camera_to_sim @ object_to_camera

    # 保存结果到新的文件,使用相同的文件名
    output_file_path = os.path.join(output_folder, filename)  # 保持文件名一致
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("物体坐标系到世界坐标系的外参 (Object to Simulation):\n")
        f.write(np.array2string(transformation_object_to_world, formatter={'float_kind': lambda x: f"{x:.6f}"}))

    print(f"文件已保存到: {output_file_path}")
    print("Object to World Transformation Matrix:")
    print(transformation_object_to_world)

