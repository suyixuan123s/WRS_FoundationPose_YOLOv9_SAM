import copy
import math
import time

import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.ag145.ag145 as dh
import robot_sim.robots.gofa5.gofa5 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr

# import robot_con.gofa_con.gofa_con as gofa_con


# def go_init():
#     init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     current_jnts = rbt_s.get_jnt_values("arm")
#
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
    # rbt_r = gofa_con.GoFaArmController()
    start_conf = np.array([0.0439823, -0.53023103, 1.05243354, 0.0143117, 1.55351757, 1.57079633])
    # rbt_r.move_j(start_conf)
    # base.run()
    # print(rbt_r.get_jnt_values())
    # rbt_s.fk("arm",np.asarray(rbt_r.get_jnt_values()))
    # rbt_s.gen_meshmodel().attach_to(base)
    # base.run()
    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)
    # go_init()

    # base.run()
    # rbt_s.hnd.open()
    # rbt_s.gen_meshmodel().attach_to(base)
    manipulator_name = "arm"
    # start_conf = rbt_s.get_jnt_values(manipulator_name)
    # start_conf = np.array([0.0439823 , -0.53023103  ,1.05243354 , 0.0143117  , 1.55351757 , 1.57079633])
    start_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(start_conf)
    hand_name = "hnd"

    # 碰撞检测列表
    obstacle_list = []

    # object
    rack_name = "rack_10ml_new"
    rack_10ml = cm.CollisionModel(f"objects/{rack_name}.stl")
    rack_10ml.set_rgba([1, 1, 1, 1])
    rack_10ml.set_pos(np.array([.6, -.5, 0]))
    rack_10ml.set_rotmat()
    rack_10ml.attach_to(base)
    obstacle_list.append(rack_10ml)

    tube_name = "blood_tube10"
    tube1 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube1.set_rgba([.0, 1, .0, 1])
    tube1.set_pos(np.array([.555, -.5, 0.113]))
    tube1.set_rotmat()
    tube1.attach_to(base)
    obstacle_list.append(tube1)

    tube2 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube2.set_rgba([.0, 1, .0, 1])
    tube2.set_pos(np.array([.585, -.5, 0.113]))
    tube2.set_rotmat()
    tube2.attach_to(base)
    obstacle_list.append(tube2)

    tube3 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube3.set_rgba([1, 0, 0, 1])
    tube3.set_pos(np.array([.615, -.5, 0.113]))
    tube3.set_rotmat()
    tube3.attach_to(base)
    obstacle_list.append(tube3)

    # tube4 = cm.CollisionModel(f"objects/{tube_name}.stl")
    # tube4.set_rgba([.0, 1, .0, 1])
    # tube4.set_pos(np.array([.645, -.5, 0.113]))
    # tube4.set_rotmat()
    # tube4.attach_to(base)

    tube5 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube5.set_rgba([.0, 1, .0, 1])
    tube5.set_pos(np.array([.645, -.475, 0.113]))
    tube5.set_rotmat()
    tube5.attach_to(base)
    obstacle_list.append(tube5)
    # base.run()

    # object
    objcm_name = "blood_tube10"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([1, 0, 0, 1])
    obj.set_pos(np.array([.615, -.5, 0.113]))
    obj.set_rotmat()
    obj.attach_to(base)

    # object_goal
    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([1, 1, 1, 1])
    obj_goal.set_pos(np.array([.645, -.5, 0.113]))
    obj_goal.set_rotmat()
    obj_goal.attach_to(base)

    # objcm_name = "rack_10ml_new"
    # obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    # obj.set_rgba([.9, .75, .35, 1])
    # obj.set_pos(np.array([.6, -.2, 0.03]))
    # obj.set_rotmat()
    # obj.attach_to(base)

    # base.run()

    # # object_goal
    # obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    # obj_goal.set_rgba([1, 1, 1, 1])
    # obj_goal.set_pos(np.array([.6, +.2, 0.03]))
    # obj_goal.set_rotmat()
    # obj_goal.attach_to(base)

    # gripper_s = dh.Dh60()
    gripper_s = dh.Ag145()
    # grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh76_grasps_blood.pickle')

    start_pos = obj.get_pos()
    start_rotmat = obj.get_rotmat()
    goal_pos = obj_goal.get_pos()

    goal_rotmat = obj_goal.get_rotmat()
    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

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
                                        obstacle_list=obstacle_list)
    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    print(conf_list)


    # robot_paths = []
    # jawwidth_paths = []
    # robot_path=[]
    # path_seg_id = []
    # for i in range(len(conf_list)-1):
    #     if jawwidth_list[i] != jawwidth_list[i+1]:
    #         path_seg_id.append(i)
    # for i in range(len(path_seg_id)):
    #     if i == 0 :
    #         robot_paths.append(conf_list[:path_seg_id[i]])
    #         jawwidth_paths.append(jawwidth_list[:path_seg_id[i]])
    #     else:
    #         robot_paths.append(conf_list[path_seg_id[i-1]:path_seg_id[i]])
    #         jawwidth_paths.append(jawwidth_list[path_seg_id[i - 1]:path_seg_id[i]])
    # print(robot_paths)
    # print(path_seg_id)

    # for path in robot_paths:

    # rbt_r.move_jntspace_path(conf_list)

    # if jawwidth_path[counter[0]]!= jawwidth_path[counter[0]+1]:
    #     rbt_r.move_jntspace_path([])
    #
    # return task.again

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
