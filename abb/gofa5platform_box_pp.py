"""

这段代码实现了一个完整的机器人抓取与放置仿真流程,包括路径规划、夹爪控制、物体交互和状态更新
主要使用 RRTConnect 算法进行路径规划,利用 Panda3D 进行可视化,并通过 Ag145 夹爪 实现抓取和放置动作

"""

import numpy as np
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.end_effectors.gripper.ag145.ag145 as dh
import robot_sim.robots.gofa5.gofa5 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import drivers.devices.dh.ag145 as dh_r
import robot_con.gofa_con.gofa_con as gofa_con

# def go_init():
#     init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     current_jnts = rbt_s.get_jnt_values("arm")
#     path = rrtc_s.plan(component_name="arm",
#                        start_conf=current_jnts,
#                        goal_conf=init_jnts,
#                        ext_dist=0.05,
#                        max_time=300)
#     rbt_r.move_jntspace_path(path)


if __name__ == '__main__':
    # 1. 初始化与环境设置
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    # 2. 初始化夹爪与机器人
    # Ag145 夹爪被初始化,并且通过 Ag145driver 类控制夹爪.gripper_r.init_gripper() 调用了初始化方法以启动夹爪
    # 机器人使用 GOFA5(gf5.GOFA5())机器人模型,并通过 GoFaArmController 来控制机器人的动作
    gripper_s = dh.Ag145()
    gripper_r = dh_r.Ag145driver()
    gripper_r.init_gripper()
    # gripper_r.jaw_to(0.0)
    # gripper_r.jaw_to(0.1)
    # base.run()
    rbt_s = gf5.GOFA5()
    rbt_r = gofa_con.GoFaArmController()

    # 3. 路径规划初始化
    rrtc_s = rrtc.RRTConnect(rbt_s)  # RRTConnect 算法用于规划机器人从起始位置到目标位置的路径.
    ppp_s = ppp.PickPlacePlanner(rbt_s)  # PickPlacePlanner 规划抓取和放置的动作路径.
    manipulator_name = "arm"
    hand_name = "hnd"

    # 4. 设置目标状态
    start_conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 设定机器人初始的关节配置 start_conf.
    # go_init()  # 将机器人从当前关节配置移动到初始化的配置.
    print("hi")

    # 5. 设置物体与目标位置
    # 创建了一个物体(obj),其模型通过 CollisionModel 载入并设置了物体的颜色和位置.物体在仿真环境中被附加并显示.
    objcm_name = "box_35"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([.9, .75, .35, 1])
    obj.set_pos(np.array([.6, -.2, 0.122]))
    # obj.set_rotmat()
    obj.attach_to(base)

    # 创建一个目标物体 obj_goal,设置目标位置并显示在仿真环境中
    obj_goal = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj_goal.set_rgba([1, 1, 1, 1])
    obj_goal.set_pos(np.array([.6, -.2 + 0.038, 0.122]))
    # obj_goal.set_rotmat()
    obj_goal.attach_to(base)
    base.run()

    # 6. 加载抓取信息  从文件中加载有关抓取信息的列表 grasp_info_list,这些信息在之后的抓取操作中会用到.
    # gripper_s = dh.Dh60()
    # grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='ag145_grasps.pickle')

    # 7. 计算起始与目标位姿   获取物体和目标物体的位置和旋转矩阵,并将它们转换为齐次变换矩阵.
    start_pos = obj.get_pos()
    start_rotmat = obj.get_rotmat()
    start_homo = rm.homomat_from_posrot(start_pos, start_rotmat)
    goal_pos = obj_goal.get_pos()
    goal_rotmat = obj.get_rotmat()

    obgl_start_homomat = rm.homomat_from_posrot(start_pos, start_rotmat)
    obgl_goal_homomat = rm.homomat_from_posrot(goal_pos, goal_rotmat)

    # conf_list, jawwidth_list, objpose_list = \
    #     ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
    #                                     objcm=obj,
    #                                     grasp_info_list=grasp_info_list,
    #                                     start_conf=start_conf,
    #                                     end_conf=start_conf,
    #                                     goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
    #                                     approach_direction_list=[None, np.array([0, 0, -1])],
    #                                     approach_distance_list=[.1] * 2,
    #                                     depart_direction_list=[np.array([0, 0, 1]), None],
    #                                     depart_distance_list=[.1] * 2)

    import pickle

    # with open("test_conf_list", "wb") as f:
    #     pickle.dump(conf_list, f)
    # with open("test_jawwidth_path", "wb") as f:
    #     pickle.dump(jawwidth_list, f)
    # with open("test_objpose_list", "wb") as f:
    #     pickle.dump(objpose_list, f)

    # 8. 加载预计算的路径从预先保存的文件中加载机器人的关节配置列表 (conf_list)、夹爪开口宽度路径 (jawwidth_list)
    # 以及物体的位姿路径 (objpose_list),这些数据用于规划和控制机器人的动作.
    with open("test_conf_list", "rb") as f:
        conf_list = pickle.load(f)
    with open("test_jawwidth_path", "rb") as f:
        jawwidth_list = pickle.load(f)
    with open("test_objpose_list", "rb") as f:
        objpose_list = pickle.load(f)

    # 9. 路径分段处理
    # 将路径分成多个段,每一段的关节配置和夹爪开口宽度相同.每当夹爪的开口宽度发生变化时,路径就会被分割成一个新的段.
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
        print("jnts = ,", rbt_r.get_jnt_values())
        print("torque = ,", rbt_r.get_torques())
        counter[1] += 1

        # if counter[1]==len(robot_paths[counter[0]]):
        if base.inputmgr.keymap["space"] is True:
            if len(robot_paths[counter[0]]) <= 2:
                gripper_r.jaw_to(jawwidth_paths[counter[0]][0] * 0.25)
                counter[0] += 1
                counter[1] = 0

            else:
                rbt_r.move_jntspace_path(robot_paths[counter[0]], wait=False)
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
