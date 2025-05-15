# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import math
import time
import keyboard
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import \
    robot_sim.end_effectors.gripper.ag145.ag145 as dh
import robot_sim.robots.gofa5.gofa5_Ag145 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr
import \
    motion.optimization_based.incremental_nik as inik
import robot_con.gofa_con.gofa_con as gofa_con
# import drivers.devices.dh.maingripper as dh_r
import drivers.devices.dh.ag145 as dh_r
import os  # 导入 os 模块，提供操作系统相关函数
# from estimater import *  # 姿态估计模块，暂时注释掉，因为我们直接读取结果
# from datareader import *  # 数据读取模块，暂时注释掉，因为我们直接读取结果
import argparse  # 导入 argparse 模块，用于解析命令行参数

from run_demo_foundationpose import estimate_pose
from suyixuan_sam.Task3_YOLOv9_Detect_SAM.ABB_Get_Masks_use_yolov9_and_sam_hjy import main as abb_main
from camera_capture import capture_images  # 导入 camera_capture 模块，用于图像捕获

# def go_init():
#     init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     current_jnts = rbt_s.get_jnt_values("arm")
#
#     path = rrtc_s.plan(component_name="arm",
#                        start_conf=current_jnts,
#                        goal_conf=init_jnts,
#                        ext_dist=0.05,
#                        max_time=300)
#     rbt_r.move_jntspace_path(path)


if __name__ == '__main__':

    # ---------------------- 参数设置 ----------------------
    # 图像捕获参数
    save_directory_rgb = r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/rgb'  # RGB 图像保存目录
    save_directory_depth = r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/depth'  # 深度图像保存目录

    # YOLOv9 和 SAM 参数
    weights_path = '/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/weights/osgj.pt'  # YOLOv9 权重文件路径
    target_labels = "j"  # 目标标签

    # 姿态估计参数
    mesh_file = '/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/mesh/textured_mesh.obj'  # 网格文件路径
    test_scene_dir = '/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514'  # 测试场景目录
    est_refine_iter = 5  # 姿态估计细化迭代次数
    track_refine_iter = 2  # 姿态跟踪细化迭代次数
    debug_level = 3  # 调试级别
    debug_dir = '/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/debug_data/514'  # 调试目录

    time_start = time.time()
    # ---------------------- 图像捕获和 YOLOv9 + SAM ----------------------
    # 捕获图像
    capture_images(save_directory_rgb, save_directory_depth, wait_time=5)  # 捕获 RGB 和深度图像
    rgb_image_path = os.path.join(save_directory_rgb, '000001.png')  # RGB 图像路径
    depth_image_path = os.path.join(save_directory_depth, '000001.png')  # 深度图像路径

    time_end1 = time.time()
    time_capture_images = time_end1 - time_start
    print(f"图像捕获和 YOLOv9 + SAM 耗时：{time_capture_images:.2f} 秒")

    # 确保图像已保存
    while not (os.path.exists(rgb_image_path) and os.path.exists(depth_image_path)):
        print("rgb或depth没有保存")  # 打印提示信息
        capture_images(save_directory_rgb, save_directory_depth, wait_time=5)  # 重新捕获图像
        time.sleep(1)  # 等待 1 秒
    print("rgb和depth都成功保存了!")  # 打印提示信息

    # 运行 YOLOv9 和 SAM
    abb_main(source=rgb_image_path, weights=weights_path, target_labels=target_labels)  # 运行 YOLOv9 和 SAM

    time_end2 = time.time()
    time_yolo_sam = time_end2 - time_end1
    print(f"YOLOv9 + SAM 耗时：{time_yolo_sam:.2f} 秒")


    # ---------------------- 姿态估计 ----------------------
    # 调用姿态估计函数，传递参数
    object_to_camera = estimate_pose(mesh_file, test_scene_dir, est_refine_iter, track_refine_iter, debug_level,
                                     debug_dir)

    print("Object to Camera Transformation Matrix:\n", object_to_camera)


    time_end3 = time.time()
    time_foundationpose = time_end3 - time_end2
    print(f"姿态估计耗时：{time_foundationpose:.2f} 秒")

    # ---------------------- Panda3D 环境设置 ----------------------
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])  # 创建 Panda3D 世界
    gm.gen_frame().attach_to(base)  # 生成坐标系并添加到世界

    rbt_s = gf5.GOFA5()  # 创建 GOFA5 机器人对象
    rbt_s.gen_meshmodel().attach_to(base)  # 生成机器人网格模型并添加到世界

    # 机器人操作时候需要打开
    # rbt_r = gofa_con.GoFaArmController()
    gripper_s = dh.Ag145()  # 创建 AG145 夹爪对象
    # grip_r = dh_r.Ag145driver(port='com3')
    # grip_r.init_gripper()

    # grip_r = dh_r.MainGripper('com3')
    # grip_r = dh_r.MainGripper('com6')

    # start_conf = np.array([0.0439823, -0.53023103, 1.05243354, 0.0143117, 1.55351757, 1.57079633])
    # rbt_r.move_j(start_conf)
    # base.run()
    # print(rbt_r.get_jnt_values())

    # rbt_s.fk("arm", start_conf)
    # rbt_s.gen_meshmodel().attach_to(base)
    # base.run()

    rrtc_s = rrtc.RRTConnect(rbt_s)  # 创建 RRT-Connect 路径规划器
    ppp_s = ppp.PickPlacePlanner(rbt_s)  # 创建抓取放置规划器

    # go_init()  # 机器人操作时候需要打开
    # base.run()

    rbt_s.hnd.open()  # 打开夹爪
    rbt_s.gen_meshmodel().attach_to(base)  # 生成机器人网格模型并添加到世界
    manipulator_name = "arm"  # 机械臂名称
    obstacle_list = []  # 障碍物列表

    start_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 起始关节配置
    print(start_conf)
    hand_name = "hnd"  # 夹爪名称

    # object
    objcm_name = "blood_tube10"  # 对象碰撞模型名称
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")  # 创建碰撞模型
    obj.set_rgba([.9, .75, .35, 1])  # 设置颜色

    # ---------------------- 坐标变换和对象放置 ----------------------

    transformation_camera_to_sim = np.array([[0.995588, -0.035402, -0.086900, 0.565000],
                                             [0.034767, -0.721015, 0.692047, -0.720000],
                                             [-0.087156, -0.692015, -0.716602, 0.755000],
                                             [0.000000, 0.000000, 0.000000, 1.000000]])  # 相机到仿真世界的变换矩阵

    # 计算物体到世界坐标系的变换矩阵
    transformation_object_to_world = transformation_camera_to_sim @ object_to_camera



    # 提取平移向量
    translation_vector = transformation_camera_to_sim[:3, 3]

    # 提取旋转矩阵
    rotation_matrix = transformation_camera_to_sim[:3, :3]

    # 设置对象的位置
    obj.set_pos(translation_vector)

    # 设置对象的旋转矩阵
    obj.set_rotmat(rotation_matrix)

    # obj.set_pos(np.array([.6, -.2, - 0.015 + 0.045]))
    # obj.set_rotmat(rm.rotmat_from_axangle([1, 0, 0], np.deg2rad(90)))
    obj.attach_to(base)
    # base.run()
    # obstacle_list.append(obj)

    # # object_goal
    # obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    # obj_goal.set_rgba([1, 1, 1, 1])
    # obj_goal.set_pos(np.array([.6, +.2, 0.03]))
    # obj_goal.set_rotmat()
    # obj_goal.attach_to(base)

    # # object_goal
    # obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    # obj_goal.set_rgba([1, 1, 1, 1])
    # obj_goal.set_pos(np.array([.6, +.2, 0.03]))
    # obj_goal.set_rotmat()
    # obj_goal.attach_to(base)

    # gripper_s = dh.Dh60()
    # gripper_s = dh.Ag145()

    # gripper_s = dh.Dh76(fingertip_type='r_76')
    # grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='ag145_grasps_blood1.pickle')

    start_pos = obj.get_pos()
    start_rotmat = obj.get_rotmat()

    goal_pos = np.array([.6, +.2, - 0.015])  # 起始位置
    goal_rotmat = np.eye(3)

    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    # ############
    grasp_cad = []
    for item in grasp_info_list:
        hnd_homomat = obgl_start_homomat.dot(rm.homomat_from_posrot(item[1], item[2]))  # 1位置2姿态，夹爪世界坐标系
        hnd_pos = hnd_homomat[:3, 3]
        hnd_rot = hnd_homomat[:3, :3]
        gripper_s.grip_at_with_jcpose(hnd_pos, hnd_rot, item[0])

        if not gripper_s.is_mesh_collided(objcm_list=[rbt_s.base_stand.lnks[0]['collision_model']]):
            # gripper_s.gen_meshmodel().attach_to(base)碰撞检测，去除从底部抓取的抓取点
            grasp_cad.append([hnd_pos, hnd_rot, hnd_homomat, item[0]])  # 0夹爪宽度
    # print("grasp_cad:", grasp_cad)
    # base.run()

    # 初始化抓取关节配置列表
    init_grasp_jnts_list = []
    approach_dis = -0.1  # 定义接近距离,通常用于计算接近位置

    # ########################## 遍历抓取候选列表中的每个项目 #######################################
    for item in grasp_cad:
        # 计算逆运动学以获得抓取位置的关节配置
        jnts = rbt_s.ik(component_name="arm",
                        tgt_pos=item[0],
                        tgt_rotmat=item[1],
                        seed_jnt_values=None,
                        max_niter=100,
                        num_seeds=2,
                        tcp_jnt_id=None,
                        tcp_loc_pos=None,
                        tcp_loc_rotmat=None,
                        toggle_debug=False)
        # 如果找到了有效的关节配置
        if jnts is not None:
            rbt_s.fk("arm", jnts)  # 使用正向运动学更新机械臂姿态

            # 检查是否与环境发生碰撞
            if not rbt_s.is_collided():
                # 计算接近位置和姿态
                # rbt_s.gen_meshmodel().attach_to(base)
                approach_pos = item[0] + approach_dis * item[1][:3, 2]  # 目标位置+接近距离*目标方向（负）
                approach_rot = item[1]
                # 计算逆运动学以获得接近位置的关节配置
                approach_jnts = rbt_s.ik(component_name="arm",
                                         tgt_pos=approach_pos,
                                         tgt_rotmat=approach_rot,
                                         seed_jnt_values=None,
                                         max_niter=100,
                                         num_seeds=2,
                                         tcp_jnt_id=None,
                                         tcp_loc_pos=None,
                                         tcp_loc_rotmat=None,
                                         toggle_debug=False)
                if approach_jnts is not None:
                    rbt_s.fk("arm", approach_jnts)
                    if not rbt_s.is_collided():
                        init_grasp_jnts_list.append([jnts, approach_jnts, approach_pos, approach_rot, item])
    print("init_grasp_jnts_list", len(init_grasp_jnts_list))

    # 1. 遍历初始抓取关节配置列表中的每个项目
    # for item in init_grasp_jnts_list:
    for path_id, item in enumerate(init_grasp_jnts_list):
        try:
            # 使用路径规划算法(如RRT-Connect)规划从起始配置到目标配置的路径
            path = rrtc_s.plan(component_name="arm",
                               start_conf=start_conf,
                               goal_conf=item[1],  # 接近关节的值
                               obstacle_list=[],
                               otherrobot_list=[],
                               ext_dist=0.2,
                               max_iter=300,
                               max_time=15.0,
                               smoothing_iterations=50,
                               animation=False)
            # print("PATH", path)

            for node in path:
                rbt_s.fk("arm", node)
                rbt_s.gen_meshmodel(rgba=[0, 1, 0, 0.1]).attach_to(base)
            gm.gen_frame(item[2], item[3]).attach_to(base)

            nx.draw(rrtc_s.roadmap, with_labels=True)
            plt.savefig(f"path_data/path{path_id}.png")
            plt.show()

            # nx.draw(rrtc_s.roadmap, with_labels=True, font_weight='bold')
            # plt.savefig('path.png')
            # plt.show()

            # 2. 生成从接近位置到目标位置的10个插值点
            lin_path = np.linspace(item[2], item[4][0], 10)
            lin_path_jnts_list = []  # 用于存储每个插值点的关节角度配置
            # 对于插值路径中的每个节点,计算其逆运动学解
            for node in lin_path:
                lin_path_jnts = rbt_s.ik(component_name="arm",
                                         tgt_pos=node,
                                         tgt_rotmat=item[3],
                                         seed_jnt_values=path[-1],  # 使用上一个路径的终点关节值
                                         max_niter=20,
                                         num_seeds=2,
                                         tcp_jnt_id=None,
                                         tcp_loc_pos=None,
                                         tcp_loc_rotmat=None,
                                         toggle_debug=False)
                lin_path_jnts_list.append(lin_path_jnts)  # 将计算出的关节角度配置添加到列表中
                rbt_s.fk("arm", lin_path_jnts)  # 使用正向运动学更新机械臂的姿态
                rbt_s.gen_meshmodel(rgba=[0, 0, 1, 0.1]).attach_to(base)

            # 3. 生成从目标位置向上移动0.2米的插值路径
            pick_path = np.linspace(item[4][0], item[4][0] + np.array([0, 0, 0.2]), 10)
            pick_path_jnts_list = []
            # 遍历插值路径中的每个节点
            for node in pick_path:
                # 计算机械臂的逆运动学解,得到关节角度
                pick_path_jnts = rbt_s.ik(component_name="arm",
                                          tgt_pos=node,
                                          tgt_rotmat=item[3],
                                          seed_jnt_values=path[-1],  # 使用路径的最后一个关节角度
                                          max_niter=20,
                                          num_seeds=2,
                                          tcp_jnt_id=None,
                                          tcp_loc_pos=None,
                                          tcp_loc_rotmat=None,
                                          toggle_debug=False)
                # 将计算得到的关节角度添加到列表中
                pick_path_jnts_list.append(pick_path_jnts)
                # 使用正向运动学更新机械臂的姿态
                rbt_s.fk("arm", pick_path_jnts)
                rbt_s.gen_meshmodel(rgba=[1, 0, 0, 0.1]).attach_to(base)

            # grip_path = rrtc_s.plan(component_name="arm",
            #                      tgt_pos=item[1] + [0, 0, 0.1],
            #                      tgt_rotmat=item[2],
            #                      seed_jnt_values=lin_path_jnts_list[-1],
            #                      max_niter=20,
            #                      num_seeds=2,  # 设置seed数量
            #                      tcp_jnt_id=None,
            #                      tcp_loc_pos=None,
            #                      tcp_loc_rotmat=None,
            #                      toggle_debug=False)
            # for node in grip_path:
            #     rbt_s.fk("arm", node)
            #     rbt_s.gen_meshmodel(rgba=[1, 0, 0, 0.1]).attach_to(base)
            break
            # time.sleep(2)
        except:
            pass

    # # 机器人操作时候需要打开
    # rbt_r.move_jntspace_path(path)
    # rbt_r.move_jntspace_path(lin_path_jnts_list)
    # grip_r.jaw_to(0)
    # time.sleep(1)
    # rbt_r.move_jntspace_path(pick_path_jnts_list)
    # rbt_r.move_jntspace_path(pick_path_jnts_list[::-1])
    # grip_r.jaw_to(0.076)
    # rbt_r.move_jntspace_path(lin_path_jnts_list[::-1])
    # rbt_r.move_jntspace_path(path[::-1])
    # grip_r.jaw_to(0.145)

    ########################
    base.run()

    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=obj,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=None,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[None, np.array([0, 0, -1])],
                                        approach_distance_list=[.1] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        depart_distance_list=[.1] * 2,
                                        obstacle_list=obstacle_list
                                        )

    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    print(conf_list)


    def update(robot_s,
               object_box,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter,
               task):
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        # if base.inputmgr.keymap["escape"] is True:
        pose = robot_path[counter[0]]
        robot_s.fk(manipulator_name, pose)
        robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_path[counter[0]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([1, 0, 0, 1])
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1
        return task.again


    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbt_s,
                                     obj,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
