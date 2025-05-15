"""
Author: Yixuan Su
Date: 2025/03/29 20:30
File: transformation_object_to_world.py
Description:
"""
import copy
import math
import time
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.gofa5.gofa5 as gf5
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    rbt_s = gf5.GOFA5()
    rbt_s.gen_meshmodel().attach_to(base)

    # object
    objcm_name = "rack_10ml_new"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([1, 0, 0, 0.5])
    obj.set_pos(np.array([.6, -.2, 0.03]))
    obj.set_rotmat(rm.rotmat_from_axangle([1, 0, 0], np.deg2rad(90)))

    # obj.set_rgba([1, 0, 0, 0.5])
    # pos = obj.set_pos(np.array([.647, -.242, .0475]))

    print("物体在世界坐标系中提前设定的位置:", obj.get_pos())
    print("物体在世界坐标系中提前设定的旋转:", obj.get_rotmat())
    obj.attach_to(base)
    # base.run()

    # obj_goal
    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([0, 0, 1, 1])
    obj_goal.set_rotmat()
    obj_goal.set_pos(np.array([0, 0, 0]))
    pos_start = obj_goal.get_pos()  # 获取物体的位置
    print("物体在自身坐标系下的位置:", obj_goal.get_pos())
    rotmat_start = obj_goal.set_rotmat()
    print("物体在自身坐标系下的旋转:", obj_goal.get_rotmat())

    transformation_object_to_world1 = np.array([[-0.99879344, 0.03514435, 0.03431099, 0.61165009],
                                                [-0.03552667, -0.03456736, -0.99877062, -0.20650541],
                                                [-0.03391515, -0.99878458, 0.03577464, 0.02998021],
                                                [0.0000000, 0.000000, 0.0000000, 1.0000000]])

    transformation_object_to_world = transformation_object_to_world1

    pos1 = transformation_object_to_world[:3, 3]
    obj_goal.set_pos(pos1)
    print("物体按照FounddationPose转换矩阵后得到的在世界坐标系中的位置", obj_goal.get_pos())
    pos_from_homomat_transform_points = rm.homomat_transform_points(transformation_object_to_world, pos_start)
    print("物体按照FounddationPose转换矩阵后得到的在世界坐标系中的位置2", obj_goal.get_pos())
    # 提取旋转矩阵
    rotation_matrix = transformation_object_to_world[:3, :3]
    obj_goal.set_rotmat(rotation_matrix)
    print("物体按照FounddationPose转换矩阵后得到的在世界坐标系中的旋转", obj_goal.get_rotmat())
    obj_goal.attach_to(base)

    # 计算位置的偏差
    difference = obj.get_pos() - obj_goal.get_pos()

    # 使用字符串格式化来输出不使用科学计数法的数值
    formatted_difference = [f"{value:.20f}" for value in difference]

    # 输出差值
    print("位置差值:", formatted_difference)

    # 获取当前和目标的旋转矩阵
    R_current = obj.get_rotmat()  # 当前旋转矩阵
    R_goal = obj_goal.get_rotmat()  # 目标旋转矩阵

    # 计算旋转偏差
    R_difference = np.dot(R_goal, R_current.T)  # 目标旋转矩阵与当前旋转矩阵的转置相乘

    # 输出旋转偏差矩阵
    print("旋转差值矩阵:\n", R_difference)

    base.run()
