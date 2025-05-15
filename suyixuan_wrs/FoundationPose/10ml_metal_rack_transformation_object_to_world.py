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
    # base.run()

    # object
    objcm_name = "rack_10ml_new_center"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.attach_to(base)

    # objcm_name = "textured_mesh1"
    # objt = cm.CollisionModel(f"objects/{objcm_name}.obj")
    # objt.attach_to(base)
    # base.run()

    obj.set_rgba([1, 0, 0, 0.5])
    pos = obj.set_pos(np.array([.8285, .0035, .0425]))
    print("物体在世界坐标系中提前设定的位置:", obj.get_pos())
    # pos1 = obj.set_pos(np.array([.647, -.242, 0]))

    # rotation_matrix = np.array([
    #     [0, 0, -1],
    #     [-1, 0, 0],
    #     [0, 1, 0]
    # ])
    # obj.set_rotmat(rotmat=rotation_matrix)
    obj_ratmat = obj.set_rotmat()
    print("物体在世界坐标系中提前设定的旋转:", obj.get_rotmat())
    # base.run()

    # obj.set_rotmat([0, 0, -1],
    #                [-1, 0, 0],
    #                [0, 1, 0])
    obj.attach_to(base)
    # base.run()

    # obj_goal
    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([0, 0, 1, 1])
    # obj.set_pos(np.array([.647, -.242, 0]))
    # pos1 = obj.set_pos(np.array([.647, -.242, .0475]))
    # rotation_matrix = np.array([
    #     [0, 0, -1],
    #     [-1, 0, 0],
    #     [0, 1, 0]
    # ])

    obj_goal.set_rotmat()
    obj_goal.set_pos(np.array([0, 0, 0]))
    # obj_goal.set_pos(np.array([0, 0, 0]))
    pos_start = obj_goal.get_pos()  # 获取物体的位置
    print("物体在自身坐标系下的位置:", obj_goal.get_pos())
    rotmat_start = obj_goal.set_rotmat()
    print("物体在自身坐标系下的旋转:", obj_goal.get_rotmat())

    # obj.attach_to(base)
    # base.run()

    # transformation_object_to_world = np.array([[0.992571, -0.033943, -0.116834, 0.648845],
    #                                            [0.055383, -0.728978, 0.682293, -0.266603],
    #                                            [-0.108328, -0.683696, -0.721682, 0.070208],
    #                                            [0.000000, 0.000000, 0.000000, 1.000000]])

    # obj_to_obj = np.array([[0, 0, -1, 0],
    #                        [0, 1, 0, 0],
    #                        [1, 0, 0, 0],
    #                        [0, 0, 0, 1]])

    # transformation_object_to_world1 = np.array([[0.98930653, -0.04889598, -0.13741332, 0.64498483],
    #                                             [0.05853337, -0.72983856, 0.68110939, -0.26111259],
    #                                             [-0.13359296, -0.68186926, -0.71917083, 0.06005253],
    #                                             [0.000000, 0.000000, 0.000000, 1.000000]])

    # transformation_object_to_world1 = np.array([[0.999828519, 0.0178198538, -0.00506263646, 0.648001869],
    #                                             [-0.0178218495, 0.999841216, -0.000370030917, -0.243567746],
    #                                             [0.00505518086, 0.000460939618, 0.999986721, 0.0449153722],
    #                                             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
    #                                              1.0000000000000000]])

    transformation_object_to_world1 = np.array([[-0.036247, 0.998753, -0.034332, 0.839609],
                                                [0.999024, 0.037082, 0.024002, 0.008355],
                                                [0.025246, -0.033428, -0.999122, 0.036492],
                                                [0.000000, 0.000000, 0.000000, 1.000000]]
                                               )

    # transformation_object_to_world = np.dot(transformation_object_to_world1, obj_to_obj)
    transformation_object_to_world = transformation_object_to_world1

    # pos = rm.homomat_transform_points(transformation_object_to_world, pos)
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

    # transformation_object_to_world11 = np.array([[0.992859, - 0.034617, - 0.114163, 0.648704],
    #                                              [0.055082, - 0.715832, 0.696097, - 0.266811],
    #                                              [-0.105818, - 0.697414, - 0.708813, 0.070026],
    #                                              [0.000000, 0.000000, 0.000000, 1.000000]])
    #
    # obj_goal11 = obj_goal.copy()
    # obj_goal11.set_rgba([0, 1, 0, 1])
    # pos11 = transformation_object_to_world11[:3, 3]
    # obj_goal11.set_pos(pos11)
    # print(pos11)
    # # 提取旋转矩阵
    # rotation_matrix11 = transformation_object_to_world11[:3, :3]
    # print(rotation_matrix11)
    # obj_goal11.set_rotmat(rotation_matrix11)
    # obj_goal11.attach_to(base)

    # 计算位置的偏差
    difference = obj.get_pos() - obj_goal.get_pos()
    # 输出差值
    print("位置差值:", difference)

    # 获取当前和目标的旋转矩阵
    R_current = obj.get_rotmat()  # 当前旋转矩阵
    R_goal = obj_goal.get_rotmat()  # 目标旋转矩阵

    # 计算旋转偏差
    R_difference = np.dot(R_goal, R_current.T)  # 目标旋转矩阵与当前旋转矩阵的转置相乘

    # 输出旋转偏差矩阵
    print("旋转差值矩阵:\n", R_difference)

    base.run()
