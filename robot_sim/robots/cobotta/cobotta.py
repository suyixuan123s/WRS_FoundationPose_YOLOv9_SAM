import os
import math
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import robot_sim.end_effectors.gripper.cobotta_gripper.cobotta_gripper as cbtg
import robot_sim.robots.robot_interface as ri
import basis.robot_math as rm
import time
import visualization.panda.world as wd
import modeling.geometric_model as gm
from robot_sim._kinematics.collision_checker import CollisionChecker


class Cobotta(ri.RobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)

        # base plate
        self.base_plate = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_plate')
        self.base_plate.jnts[1]['loc_pos'] = np.array([0, 0, 0.035])
        self.base_plate.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base_plate.stl")
        self.base_plate.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_plate.reinitialize()

        # arm
        arm_homeconf = np.zeros(6)
        arm_homeconf[1] = -math.pi / 6
        arm_homeconf[2] = math.pi / 2
        arm_homeconf[4] = math.pi / 6
        self.arm = cbta.CobottaArm(pos=self.base_plate.jnts[-1]['gl_posq'],
                                   rotmat=self.base_plate.jnts[-1]['gl_rotmatq'],
                                   homeconf=arm_homeconf,
                                   name='arm', enable_cc=False)

        # gripper
        self.hnd = cbtg.CobottaGripper(pos=self.arm.jnts[-1]['gl_posq'],
                                       rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                       name='hnd_s', enable_cc=False)

        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat

        # a list of detailed information about objects in hand, see
        # CollisionChecker.add_objinhnd
        # 关于对象的详细信息列表,请参见CollisionChecker.add_objinhnd
        self.oih_infos = []

        # collision detection
        if enable_cc:
            self.enable_cc()

        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_plate, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.jlc, [0, 1, 2])
        activelist = [self.base_plate.lnks[0],
                      self.arm.lnks[0],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.jlc.lnks[0],
                      self.hnd.jlc.lnks[1],
                      self.hnd.jlc.lnks[2]]
        self.cc.set_active_cdlnks(activelist)

        fromlist = [self.base_plate.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.base_plate.lnks[0],
                    self.arm.lnks[1]]
        intolist = [self.hnd.jlc.lnks[0],
                    self.hnd.jlc.lnks[1],
                    self.hnd.jlc.lnks[2]]
        self.cc.set_cdpair(fromlist, intolist)

        # TODO is the following update needed?
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_plate.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_plate.jnts[-1]['gl_posq'], rotmat=self.base_plate.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='cobotta_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_plate.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_shuidi_mobile_meshmodel',
                      option='full'):
        """
        :param tcp_jnt_id: 末端执行器(TCP,Tool Center Point)关节的 ID,用于定位和旋转.
        :param tcp_loc_pos: TCP 的位置,通常是一个三维坐标.
        :param tcp_loc_rotmat: TCP 的旋转矩阵,用于定义其方向.
        :param toggle_tcpcs: 一个布尔值,指示是否显示 TCP.
        :param toggle_jntscs: 一个布尔值,指示是否显示关节坐标系.
        :param rgba: 一个包含颜色和透明度信息的数组(红、绿、蓝、透明度).
        :param name: 生成的网格模型的名称,默认为 'xarm_shuidi_mobile_meshmodel'.
        :param option: 一个字符串,指示生成模型的类型,可以是 'full'、'hand_only' 或 'body_only'.

        如果 option 是 'full' 或 'body_only',则生成基础板和机械臂的模型,并将其附加到 meshmodel.
        如果 option 是 'full' 或 'hand_only',则生成末端执行器(hand)的模型,并将其附加到 meshmodel.
        如果 option 是 'full',则遍历 self.oih_infos 中的对象信息,复制每个对象的碰撞模型,设置其位置和旋转,并附加到 meshmodel.
        :return:
        """
        meshmodel = mc.ModelCollection(name=name)
        if option == 'full':
            # 生成完整的模型(主体、手部、附加对象等)
            self.base_plate.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                          toggle_tcpcs=False,
                                          toggle_jntscs=toggle_jntscs,
                                          rgba=rgba).attach_to(meshmodel)
            self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            self.hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
            for obj_info in self.oih_infos:
                objcm = obj_info['collision_model'].copy()
                objcm.set_pos(obj_info['gl_pos'])
                objcm.set_rotmat(obj_info['gl_rotmat'])
                if rgba is not None:
                    objcm.set_rgba(rgba)
                objcm.attach_to(meshmodel)

        elif option == 'body_only':
            # 生成主体模型
            self.base_plate.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                          toggle_tcpcs=False,
                                          toggle_jntscs=toggle_jntscs,
                                          rgba=rgba).attach_to(meshmodel)
            self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)

        elif option == 'hand_only':
            # 生成手部模型
            self.hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)
    robot_s = Cobotta(enable_cc=True)
    robot_s.jaw_to(.02)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=True, option='body_only').attach_to(base)
    # robot_s.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(base)
    robot_s.show_cdprimit()
    # base.run()

    tgt_pos = np.array([.25, 0, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    robot_s.show_cdprimit()
    # base.run()

    component_name = 'arm'
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    print("jnt_values:", jnt_values)
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True, option='body_only')
    robot_s_meshmodel.attach_to(base)
    robot_s.show_cdprimit()
    # base.run()

    robot_s.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = robot_s.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()
