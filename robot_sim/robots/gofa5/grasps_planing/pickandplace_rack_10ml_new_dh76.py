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
import robot_sim.end_effectors.gripper.dh76.dh76 as dh
import robot_sim.robots.gofa5.gofa5_dh76 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
from direct.task.TaskManagerGlobal import taskMgr
import motion.optimization_based.incremental_nik as inik
# import robot_con.gofa_con.gofa_con as gofa_con
import drivers.devices.dh.maingripper as dh_r

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

    # # 机器人操作时候需要打开
    # rbt_r = gofa_con.GoFaArmController()
    # grip_r = dh_r.MainGripper('com3')

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

    # object
    objcm_name = "rack_10ml_new"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([.9, .75, .35, 1])
    obj.set_pos(np.array([.6, -.2, - 0.015 + 0.045]))
    obj.set_rotmat(rm.rotmat_from_axangle([1, 0, 0], np.deg2rad(90)))
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

    gripper_s = dh.Dh76(fingertip_type='r_76')
    # grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh76_grasps_rack_10ml_new_r_76.pickle')

    start_pos = obj.get_pos()
    start_rotmat = obj.get_rotmat()

    goal_pos = np.array([.6, +.2, - 0.015])
    goal_rotmat = np.eye(3)

    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    # ############
    grasp_cad = []
    for item in grasp_info_list:
        hnd_homomat = obgl_start_homomat.dot(rm.homomat_from_posrot(item[1], item[2]))
        hnd_pos = hnd_homomat[:3, 3]
        hnd_rot = hnd_homomat[:3, :3]
        gripper_s.grip_at_with_jcpose(hnd_pos, hnd_rot, item[0])

        if not gripper_s.is_mesh_collided(objcm_list=[rbt_s.base_stand.lnks[0]['collision_model']]):
            # gripper_s.gen_meshmodel().attach_to(base)
            grasp_cad.append([hnd_pos, hnd_rot, hnd_homomat, item[0]])
    print("grasp_cad:", grasp_cad)
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
                approach_pos = item[0] + approach_dis * item[1][:3, 2]
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
            print("PATH", path)

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

    # 机器人操作时候需要打开
    # rbt_r.move_jntspace_path(path)
    # rbt_r.move_jntspace_path(lin_path_jnts_list)
    # grip_r.jaw_to(0)
    # time.sleep(1)
    # rbt_r.move_jntspace_path(pick_path_jnts_list)
    # rbt_r.move_jntspace_path(pick_path_jnts_list[::-1])
    # grip_r.jaw_to(0.076)
    # rbt_r.move_jntspace_path(lin_path_jnts_list[::-1])
    # rbt_r.move_jntspace_path(path[::-1])

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
