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
# import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.end_effectors.gripper.ag145.ag145 as dh
# import robot_sim.end_effectors.gripper.dh76.dh76 as dh
import robot_sim.robots.gofa5.gofa5_Ag145 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr
import motion.optimization_based.incremental_nik as inik
import robot_con.gofa_con.gofa_con as gofa_con
# import drivers.devices.dh.maingripper as dh_r
import drivers.devices.dh.ag145 as dh_r

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
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    rbt_s = gf5.GOFA5()
    rbt_s.gen_meshmodel().attach_to(base)

    # 机器人操作时候需要打开
    # rbt_r = gofa_con.GoFaArmController()
    gripper_s = dh.Ag145()
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

    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)

    # go_init()  # 机器人操作时候需要打开
    # base.run()

    rbt_s.hnd.open()
    rbt_s.gen_meshmodel().attach_to(base)
    manipulator_name = "arm"
    obstacle_list = []

    start_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(start_conf)
    hand_name = "hnd"

    # object_goal
    objcm_name = "rack_10ml_new_center"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([1, 1, 1, 1])
    obj.set_pos(np.array([.6, +.2, 0.06]))
    obj.set_rotmat()
    obj.attach_to(base)

    # # object_goal
    # obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    # obj_goal.set_rgba([1, 1, 1, 1])
    # obj_goal.set_pos(np.array([.6, -.2, 0.03]))
    # obj_goal.set_rotmat()
    # obj_goal.attach_to(base)
    # # base.run()

    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='ag145_grasps_rack_10ml_new_center.pickle')

    print("grasp_info_list的长度", len(grasp_info_list))
    start_pos = obj.get_pos()
    start_rotmat = obj.get_rotmat()

    goal_pos = np.array([.6, -.2, 0.06])
    goal_rotmat = np.eye(3)
    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=obj,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=None,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[np.array([0, 0, -1]), np.array([0, 0, -1])],
                                        approach_distance_list=[.1],
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        depart_distance_list=[.1],
                                        obstacle_list=obstacle_list
                                        )

    print("conf_list", conf_list)
    print("jawwidth_list", jawwidth_list)
    print("objpose_list", objpose_list)
    base.run()

    robot_attached_list = []
    object_attached_list = []
    counter = [0]


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
