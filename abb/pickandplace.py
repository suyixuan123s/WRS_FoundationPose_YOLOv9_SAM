import math
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.gofa5.gofa5 as gf5
import modeling.collision_model as cm
import numpy as np
import grasping.planning.antipodal as gpa
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm
import robot_sim.end_effectors.gripper.ag145.ag145 as ag
import robot_sim.end_effectors.gripper.ag145.ag145 as dh


def ava_space(obj_poslist, step=0.08, slight_center=[0.54, 0, 0]):
    ##可用的空间  z默认为0
    ava_space = []
    for i in range(-5, 6, 1):
        for j in range(-5, 6, 1):
            [x, y, z] = [slight_center[0] + step * i, slight_center[1] + step * j, slight_center[2]]
            [x, y, z] = np.round([x, y, z], 2)
            ava_space.append([x, y, z])
    for obj_xyz in obj_poslist:
        ava_space.remove(obj_xyz)
    return ava_space


def update_obj_poslist(obj_list):
    new_obj_poslist = []
    for obj in obj_list:
        obj_pos = obj.get_pos()
        obj_pos[2] = 0
        obj_pos = np.round(obj_pos, 2)
        obj_pos = obj_pos.tolist()
        new_obj_poslist.append(obj_pos)
    print("新的obj列表？", new_obj_poslist)
    return new_obj_poslist


def rand_xyz(obj, step=0.08, slight_center=[0.54, 0, 0], ):
    rand_int = np.random.randint(-5, 5, 2)
    rand_x = slight_center[0] + rand_int[0] * step
    rand_y = slight_center[0] + rand_int[1] * step
    z = obj.get_pos()[2]
    return np.array([rand_x, rand_y, z])


def grasper_plannning(obj, angle_between_contact_normals=math.radians(177), max_samples=5):
    global gripper_s
    gripper_s = ag.Ag145()
    grasp_info_list = gpa.plan_grasps(gripper_s, obj, angle_between_contact_normals,
                                      openning_direction="loc_y",
                                      max_samples=max_samples,
                                      min_dist_between_sampled_contact_points=0.003,
                                      contact_offset=0.003)
    return grasp_info_list


##6个面的长方体
def rand_changfangti(init_obj, rand_pos):
    all_rotmat = []
    all_rotmat.append(np.eye(3))
    rot_axis = [[1, 0, 0], [0, 1, 0]]
    rot_angel = [math.pi / 2, math.pi, 3 * math.pi / 2]
    for axis in rot_axis:
        for angel in rot_angel:
            rotmatxy = rm.rotmat_from_axangle(axis=axis, angle=angel)
            print("单独的？: ", rotmatxy)
            all_rotmat.append(rotmatxy)
    print("所有的", all_rotmat)
    i = np.random.randint(len(all_rotmat))
    print("选中了i", i)
    ##设置关于世界z轴旋转: 
    angel_z = np.random.uniform(0, 2 * math.pi)
    print("绕世界z旋转: ", angel_z)
    rotmat = np.dot(rm.rotmat_from_axangle(axis=[0, 0, 1], angle=angel_z), all_rotmat[i])
    print(rotmat)
    obj = init_obj.copy()
    obj.set_pos(rand_pos)
    obj.set_rotmat(rotmat)
    obj.set_rgba([0, 1, 0, 0.5])
    obj.attach_to(base)
    gm.gen_frame(rand_pos, obj.get_rotmat(), length=0.15).attach_to(base)
    return obj


if __name__ == '__main__':

    base = wd.World(cam_pos=[2.554, 2.5, 2.5], lookat_pos=[0, 0, 0])  ##看向的位置
    gm.gen_frame().attach_to(base)

    # 障碍物,obj定义
    obstacle_list = []
    obj_list = []
    ##定义目标位置 和视线
    # slight = gm.movegmbox(extent=[0.8, 0.8, 0.005], rgba=[1, 0, 0, 0.6], pos=[0.54, 0, 0.01])
    # ##    extent---总长  坐标为一半
    # slight.attach_to(base)
    slight_center = [0.54, 0, 0]
    gm.gen_frame(slight_center).attach_to(base)

    ## 测试一下slight的四个点坐标
    ball1 = cm.gen_sphere(radius=0.01, rgba=[1, 0, 0, 1])
    ball1.set_pos(np.array([0.14, -0.4, 0]))
    ball1.attach_to(base)

    ball2 = cm.gen_sphere(radius=0.01, rgba=[0, 1, 0, 1])
    ball2.set_pos(np.array([0.14, 0.4, 0]))
    ball2.attach_to(base)

    ball3 = cm.gen_sphere(radius=0.01, rgba=[0, 0, 1, 1])
    ball3.set_pos(np.array([0.94, -0.4, 0]))
    ball3.attach_to(base)

    ball4 = cm.gen_sphere(radius=0.01, rgba=[1, 1, 1, 1])
    ball4.set_pos(np.array([0.94, 0.4, 0]))
    ball4.attach_to(base)

    # 定义桌子碰撞体
    # table = cm.gen_box(extent=[1.22, 1.5, 0.8], rgba=[0, 0, 1, 0.1])
    # pos_table = [0.50, 0.0, -0.41]
    # table.set_pos(np.array(pos_table))
    # table.attach_to(base)
    # table.show_cdmesh()
    # obstacle_list.append(table)
    #
    # ##定义相机架碰撞体
    # gan = cm.gen_box(extent=[0.1, 0.15, 2.5], rgba=[0, 0, 1, 0.5])
    # pos_gan = [0.54, 0.78, 0.25]
    # gan.set_pos(np.array(pos_gan))
    # gan.attach_to(base)
    # gan.show_cdprimit()
    # kuai = cm.gen_box(extent=[0.08, 0.21, 0.06], rgba=[0, 0, 1, 0.5])
    # pos_kuai = [0.54, 0.65, 0.015]
    # kuai.set_pos(np.array(pos_kuai))
    # kuai.attach_to(base)
    # kuai.show_cdprimit()
    # xiangji = cm.gen_box(extent=[0.35, 0.15, 0.15], rgba=[0, 0, 1, 0.5])
    # pos_xiangji = [0.52, 0.65, 1.3]
    # xiangji.set_pos(np.array(pos_xiangji))
    # xiangji.attach_to(base)
    # xiangji.show_cdprimit()
    # obstacle_list.append(xiangji)
    # obstacle_list.append(gan)
    # obstacle_list.append(kuai)

    # np.random.seed(10)
    ## 物体obj
    init_posobj = ava_space(obj_poslist=[]).copy()  ##whole space
    for init_pos in init_posobj:
        init_pos[2] = 0.1

    ##原固定的
    # obj1=cm.gen_box(extent=[0.04, 0.08, 0.1], rgba=[0, 0, 1, 1])
    # obj1.set_pos(pos=np.array(init_posobj[45]))
    # obj1.attach_to(base)
    #

    ##随机的原obj
    obj1 = cm.CollisionModel(
        r'E:\ABB-Project\ABB_wrs\abb\objects\tixing.STL')  ##\是linux系统路径  /windows系统  r-将linux转为windows
    obj1.set_rgba([1, 1, 0, 1])
    homomat = np.array([[8.00000030e-01, -3.99999978e-01, 4.47213562e-01,
                         -3.67082076e-02],
                        [-3.99999978e-01, 1.99999970e-01, 8.94427208e-01,
                         1.58359297e-03],
                        [-4.47213562e-01, -8.94427208e-01, -4.44089226e-17,
                         6.70820378e-02],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         1.00000000e+00]])
    # obj1.attach_to(base)
    gm.gen_frame(pos=obj1.get_pos(), rotmat=obj1.get_rotmat()).attach_to(base)
    # rotmat_temp=rm.rotmat_from_axangle(axis=[0,0,1],angle=math.degrees(5))
    # homomat_temp=rm.homomat_from_posrot(rot=rotmat_temp)
    # homomat=np.dot(homomat_temp,homomat1)

    print("看看homomat", homomat)
    obj1.set_homomat(homomat)
    obj1.set_pos(init_posobj[45])
    print("youbianhau ?", obj1.get_homomat())
    obj1.attach_to(base)

    ##目标
    # obj2=cm.gen_box(extent=[0.04, 0.08, 0.1], rgba=[0, 0, 1, 0.5])
    # obj2_pos=init_posobj[85]
    #
    # obj2.set_pos(np.array(obj2_pos))
    # obj2.set_rotmat(np.array(rm.rotmat_from_axangle(axis=[1,0,0],angle=math.pi/2)))
    # obj2.attach_to(base)
    # gm.gen_frame(obj2.get_pos(),obj2.get_rotmat()).attach_to(base)

    obj2 = obj1.copy()
    obj2.set_rgba([0, 1, 0, 1])
    obj2.set_homomat(homomat)
    obj2.set_pos(init_posobj[72])
    obj2.attach_to(base)
    # base.run()
    # 更新空间
    # print("现在的obj_list",len(obj_list))
    # obj_poslist=update_obj_poslist(obj_list)
    # avail_space=ava_space(obj_poslist=obj_poslist)
    # print("可用空间空间为？",avail_space)

    ##obj0放置的目标obj00位置
    # for avail_pos in avail_space:
    #     avail_pos[2]=0.05
    # obj00 =obj2.copy()
    # obj00.set_rgba([0,0,1,0.5])
    # obj00.set_pos(np.array([0.7,0.35,0.06]))
    # print("start_pos:",obj1.get_pos())
    # print("goal-pos", obj00.get_pos())
    # obj00.attach_to(base)

    # np.random.seed(0)
    # obj_list.append(obj0)
    # obj_list.append(obj1)
    # obj_list.append(obj2)
    # np.random.rand(1)

    rg_s = gf5.GOFA5()

    gm.gen_frame(rg_s.get_gl_tcp("hnd")[0], rg_s.get_gl_tcp("hnd")[1]).attach_to(base)
    rg_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    rg_s.show_cdprimit()
    rrtc_s = rrtc.RRTConnect(rg_s)
    ppp_s = ppp.PickPlacePlanner(rg_s)
    start_conf = np.array([0, 0, 0, 0, 0, 0])
    base.run()
    ##读取爪子: 
    gripper_s = dh.Ag145()

    grasp_info_list = gpa.load_pickle_file(objcm_name="tixing", root=None,
                                           file_name="objti_grasptest_250108.pickle")  ##294
    print(len(grasp_info_list))  ##---box0  274
    # base.run()

    start_homo = np.array(obj1.get_homomat())

    # gm.gen_frame(start_pos,start_rotmat).attach_to(base)##   末端执行器和物体是不一样的

    end_pos = np.array(obj2.get_pos())
    end_rotmat = np.array(obj2.get_rotmat())
    end_homo = rm.homomat_from_posrot(end_pos, end_rotmat)

    # ##检查是否能达到  start  和  end
    #     jnt_values=rg_s.ik(component_name="arm",tgt_pos=start_pos,tgt_rotmat=start_rotmat)
    #     rg_s.fk(component_name="arm",jnt_values=jnt_values)   ## start 能到到
    #     rg_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    #     # base.run()
    #
    #
    #     jnt_values=rg_s.ik(component_name="arm",tgt_pos=end_pos,tgt_rotmat=end_rotmat)
    #     rg_s.fk(component_name="arm",jnt_values=jnt_values)   ## end 能到到
    #     rg_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    #     base.run()
    obstacle_list = []

    conf_list, jawwidth_list, objpose_list = \
        ppp_s.gen_pick_and_place_motion(hnd_name="hnd",
                                        objcm=obj1,
                                        grasp_info_list=grasp_info_list,
                                        start_conf=start_conf,
                                        end_conf=None,
                                        goal_homomat_list=[start_homo, end_homo],
                                        approach_direction_list=[None, np.array([0, 0, -1])],
                                        approach_distance_list=[.1] * 2,
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        obstacle_list=obstacle_list,
                                        depart_distance_list=[.1] * 2)

    robot_attached_list = []
    object_attached_list = []
    counter = [0]
    print("长度是？？", len(conf_list))
    print("k看看你conf", conf_list)


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
        robot_s.fk("arm", pose)
        robot_s.jaw_to("hnd", jawwidth_path[counter[0]])
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
                          extraArgs=[rg_s,
                                     obj1,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()
