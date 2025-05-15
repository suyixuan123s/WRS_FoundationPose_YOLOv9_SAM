import copy
import math
import time
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.end_effectors.gripper.ag145.ag145 as dh
import robot_sim.robots.gofa5.gofa5_Ag145 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import drivers.devices.dh.ag145 as dh_r
import robot_con.gofa_con.gofa_con as gofa_con

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

    gripper_r = dh.Ag145()
    # gripper_r = dh_r.Ag145driver()
    # gripper_r.init_gripper()
    # gripper_r.jaw_to(0.0)
    # # time.sleep(3)
    # gripper_r.jaw_to(0.1)
    # # time.sleep(3)
    # gripper_r.jaw_to(0.05)
    # base.run()

    # print("3")
    rbt_s = gf5.GOFA5()
    rbt_s.gen_meshmodel().attach_to(base)

    # rbt_r = gofa_con.GoFaArmController()

    rrtc_s = rrtc.RRTConnect(rbt_s)
    ppp_s = ppp.PickPlacePlanner(rbt_s)
    manipulator_name = "arm"
    hand_name = "hnd"

    start_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # go_init()
    # print("hi")

    # object
    objcm_name = "rack_10ml_new"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([1, 1, 1, 1])
    obj.set_pos(np.array([.6, .2, -0.015]))
    # obj.set_rotmat()
    obj.attach_to(base)

    # object_goal
    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([1, 1, 1, 1])
    obj_goal.set_pos(np.array([.6, -.2 + 0.038, -0.015]))
    # obj_goal.set_rotmat()
    obj_goal.attach_to(base)

    # base.run()

    # gripper_s = dh.Dh60()

    # grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='ag145_grasps.pickle')

    start_pos = obj.get_pos()
    start_rotmat = obj.get_rotmat()

    # start_homo = rm.homomat_from_posrot(start_pos, start_rotmat)
    goal_pos = obj_goal.get_pos()
    goal_rotmat = obj.get_rotmat()

    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    approach_jawwidth = ppp_s.robot_s.hnd_dict["hnd"].jawwidth_rng[1]

    # 获取第一个目标位置和旋转矩阵
    first_goal_pos = start_pos
    first_goal_rotmat = start_rotmat

    # 获取最后一个目标位置和旋转矩阵
    last_goal_pos = goal_pos
    last_goal_rotmat = goal_rotmat

    objcm_list=[]
    goal_homomat_list = [obgl_start_homomat, obgl_goal_homomat]
    objcm_list.append(rbt_s.base_stand.lnks[0]['collision_model'])

    common_grasp_id_list, _ = ppp_s.find_common_graspids(hand_name='hnd',
                                                         grasp_info_list=grasp_info_list,
                                                         goal_homomat_list=goal_homomat_list,
                                                         obstacle_list=[])

    if len(common_grasp_id_list) is not None:
        print("找到与目标位姿匹配的抓取 IDcommon_grasp_id_list的长度", len(common_grasp_id_list))
    # 如果没有找到任何匹配的抓取ID,打印错误信息并返回
    elif len(common_grasp_id_list) == 0:
        print("No common grasp id at the given goal homomats!")

    for grasp_id in common_grasp_id_list:
        grasp_info = grasp_info_list[grasp_id]

        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info

        # 计算接近时的夹爪中心位置和旋转矩阵
        first_jaw_center_pos = first_goal_rotmat.dot(jaw_center_pos) + first_goal_pos
        first_jaw_center_rotmat = first_goal_rotmat.dot(jaw_center_rotmat)

        # 将物体控制模型复制并设置为障碍物
        objcm_copy = obj.copy()  # 复制物体控制模型
        objcm_copy.set_pos(first_goal_pos)  # 设置物体位置
        objcm_copy.set_rotmat(first_goal_rotmat)  # 设置物体旋转矩阵
        conf_list_approach, jawwidthlist_approach = ppp_s.gen_approach_motion(component_name="hnd",
                                                                              goal_tcp_pos=first_jaw_center_pos,
                                                                              goal_tcp_rotmat=first_jaw_center_rotmat,
                                                                              start_conf=start_conf,
                                                                              approach_direction=None,
                                                                              approach_distance=[.1],
                                                                              approach_jawwidth=approach_jawwidth,
                                                                              obstacle_list=[],
                                                                              # object_list=[objcm_copy],  # 将物体作为障碍物
                                                                              object_list=[],  # 将物体作为障碍物
                                                                              seed_jnt_values=start_conf)

        if conf_list_approach is not None:
            print("conf_list_approach", conf_list_approach)
            print("len(conf_list_approach)", len(conf_list_approach))
            print("jawwidthlist_approach", jawwidthlist_approach)
            # 如果无法生成接近运动,打印错误信息并继续下一个抓取ID
            seed_conf = conf_list_approach[-1]
            rbt_s.fk('arm', seed_conf)
            rbt_s.gen_meshmodel().attach_to(base)
        elif conf_list_approach is None:
             continue

    base.run()

    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                        objcm=obj,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                        approach_direction_list=[None, np.array([0, 0, -1])],
                                        approach_distance_list=[.1] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        depart_distance_list=[.1] * 2)

    import pickle

    # with open("test_conf_list", "wb") as f:
    #     pickle.dump(conf_list, f)
    # with open("test_jawwidth_path", "wb") as f:
    #     pickle.dump(jawwidth_list, f)
    # with open("test_objpose_list", "wb") as f:
    #     pickle.dump(objpose_list, f)

    with open("test_conf_list", "rb") as f:
        conf_list = pickle.load(f)
    with open("test_jawwidth_path", "rb") as f:
        jawwidth_list = pickle.load(f)
    with open("test_objpose_list", "rb") as f:
        objpose_list = pickle.load(f)

    robot_paths = []
    jawwidth_paths = []
    objposes_list = []
    robot_path = []
    path_seg_id = [0]

    for i in range(len(conf_list) - 1):
        if jawwidth_list[i] != jawwidth_list[i - 1]:
            path_seg_id.append(i)
            path_seg_id.append(i + 1)
            # path_seg_id.append(i + 2)
    path_seg_id.append(len(conf_list))

    for i in range(len(path_seg_id) - 1):
        robot_paths.append(conf_list[path_seg_id[i]:path_seg_id[i + 1]])
        jawwidth_paths.append(jawwidth_list[path_seg_id[i]:path_seg_id[i + 1]])
        objposes_list.append(objpose_list[path_seg_id[i]:path_seg_id[i + 1]])
    # print(robot_paths)
    # print(path_seg_id)

    # for path in robot_paths:

    robot_attached_list = []
    object_attached_list = []
    counter = [0, 0]
    print(conf_list)
    base.run()


    def update(robot_s,
               object_box,
               robot_paths,
               jawwidth_paths,
               obj_paths,
               robot_attached_list,
               object_attached_list,
               counter,
               task):

        if counter[0] >= len(robot_paths):
            counter[0] = 0

        if counter[1] >= len(robot_paths[counter[0]]):
            counter[1] = 0

        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()

            robot_attached_list.clear()
            object_attached_list.clear()

        pose = robot_paths[counter[0]][counter[1]]
        robot_s.fk(manipulator_name, pose)
        robot_s.jaw_to(hand_name, jawwidth_paths[counter[0]][counter[1]])
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_paths[counter[0]][counter[1]]
        objb_copy = object_box.copy()
        objb_copy.set_rgba([1, 0, 0, 1])
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[1] += 1
        # if counter[1]==len(robot_paths[counter[0]]):
        if base.inputmgr.keymap["space"] is True:
            if len(robot_paths[counter[0]]) <= 2:
                gripper_r.jaw_to(jawwidth_paths[counter[0]][0] * 0.25)
                counter[0] += 1
                counter[1] = 0
                time.sleep(1)
            else:
                rbt_r.move_jntspace_path(robot_paths[counter[0]])
                counter[0] += 1
                counter[1] = 0

        return task.again


    taskMgr.doMethodLater(0.05, update, "update",
                          extraArgs=[rbt_s,
                                     obj,
                                     robot_paths,
                                     jawwidth_paths,
                                     objposes_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
