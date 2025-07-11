import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import modeling.collision_model as cm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.irb14050.irb14050 as ya
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim.robots.robot_interface as ri


class Yumi(ri.RobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)

        # lft
        self.lft_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(7), name='lft_body')
        self.lft_body.jnts[1]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[2]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[3]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[4]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[5]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[6]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[7]['loc_pos'] = np.array([0, 0, 0])

        self.lft_body.jnts[8]['loc_pos'] = np.array([0.05355, 0.07250, 0.41492])
        self.lft_body.jnts[8]['loc_rotmat'] = rm.rotmat_from_euler(0.9781, -0.5716, 2.3180)  # left from robot_s view

        self.lft_body.lnks[0]['name'] = "yumi_lft_stand"
        self.lft_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_tablenotop.stl")
        self.lft_body.lnks[0]['rgba'] = [.55, .55, .55, 1.0]

        self.lft_body.lnks[1]['name'] = "yumi_lft_body"
        self.lft_body.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[1]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "body.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)

        self.lft_body.lnks[1]['rgba'] = [.7, .7, .7, 1]

        self.lft_body.lnks[2]['name'] = "yumi_lft_column"
        self.lft_body.lnks[2]['loc_pos'] = np.array([-.327, -.24, -1.015])
        self.lft_body.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_column60602100.stl")
        self.lft_body.lnks[2]['rgba'] = [.35, .35, .35, 1.0]

        self.lft_body.lnks[3]['name'] = "yumi_lft_column"
        self.lft_body.lnks[3]['loc_pos'] = np.array([-.327, .24, -1.015])
        self.lft_body.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_column60602100.stl")
        self.lft_body.lnks[3]['rgba'] = [.35, .35, .35, 1.0]

        self.lft_body.lnks[4]['name'] = "yumi_top_back"
        self.lft_body.lnks[4]['loc_pos'] = np.array([-.327, 0, 1.085])
        self.lft_body.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[4]['rgba'] = [.35, .35, .35, 1.0]

        self.lft_body.lnks[5]['name'] = "yumi_top_lft"
        self.lft_body.lnks[5]['loc_pos'] = np.array([-.027, -.24, 1.085])
        self.lft_body.lnks[5]['loc_rotmat'] = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self.lft_body.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[5]['rgba'] = [.35, .35, .35, 1.0]

        self.lft_body.lnks[6]['name'] = "yumi_top_lft"
        self.lft_body.lnks[6]['loc_pos'] = np.array([-.027, .24, 1.085])
        self.lft_body.lnks[6]['loc_rotmat'] = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self.lft_body.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[6]['rgba'] = [.35, .35, .35, 1.0]

        self.lft_body.lnks[7]['name'] = "yumi_top_front"
        self.lft_body.lnks[7]['loc_pos'] = np.array([.273, 0, 1.085])
        self.lft_body.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "yumi_column6060540.stl")
        self.lft_body.lnks[7]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.reinitialize()

        lft_arm_homeconf = np.radians(np.array([20, -90, 120, 30, 0, 40, 0]))
        self.lft_arm = ya.IRB14050(pos=self.lft_body.jnts[-1]['gl_posq'],
                                   rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                   homeconf=lft_arm_homeconf, enable_cc=False)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'], rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        self.lft_hnd = yg.YumiGripper(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                      rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                      enable_cc=False, name='lft_hnd')
        self.lft_hnd.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'], rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])

        # rgt
        self.rgt_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='rgt_body')
        self.rgt_body.jnts[1]['loc_pos'] = np.array([0.05355, -0.0725, 0.41492])
        self.rgt_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(-0.9795, -0.5682, -2.3155)  # left from robot_s view
        self.rgt_body.lnks[0]['name'] = "yumi_rgt_body"
        self.rgt_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.rgt_body.reinitialize()

        rgt_arm_homeconf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
        self.rgt_arm = self.lft_arm.copy()
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'], rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        self.rgt_arm.set_homeconf(rgt_arm_homeconf)
        self.rgt_arm.goto_homeconf()
        self.rgt_hnd = self.lft_hnd.copy()
        self.rgt_hnd.name = 'rgt_hnd'
        self.rgt_hnd.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'], rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])

        # tool center point
        # lft
        self.lft_arm.tcp_jnt_id = -1
        self.lft_arm.tcp_loc_pos = self.lft_hnd.jaw_center_pos
        self.lft_arm.tcp_loc_rotmat = self.lft_hnd.jaw_center_rotmat

        # rgt
        self.rgt_arm.tcp_jnt_id = -1
        self.rgt_arm.tcp_loc_pos = self.rgt_hnd.jaw_center_pos
        self.rgt_arm.tcp_loc_rotmat = self.rgt_hnd.jaw_center_rotmat

        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []

        # collision detection
        if enable_cc:
            self.enable_cc()

        # component map
        self.manipulator_dict['rgt_arm'] = self.rgt_arm
        self.manipulator_dict['lft_arm'] = self.lft_arm
        self.manipulator_dict['rgt_hnd'] = self.rgt_arm
        self.manipulator_dict['lft_hnd'] = self.lft_arm
        self.hnd_dict['rgt_hnd'] = self.rgt_hnd
        self.hnd_dict['lft_hnd'] = self.lft_hnd
        self.hnd_dict['rgt_arm'] = self.rgt_hnd
        self.hnd_dict['lft_arm'] = self.lft_hnd

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.2, 0, 0.04),
                                              x=.16 + radius, y=.2 + radius, z=.04 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.24, 0, 0.24),
                                              x=.12 + radius, y=.125 + radius, z=.24 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.07, 0, 0.4),
                                              x=.075 + radius, y=.125 + radius, z=.06 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0, 0.145, 0.03),
                                              x=.135 + radius, y=.055 + radius, z=.03 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0, -0.145, 0.03),
                                              x=.135 + radius, y=.055 + radius, z=.03 + radius)
        collision_node.addSolid(collision_primitive_r0)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.lft_body, [0, 1, 2, 3, 4, 5, 6, 7])
        self.cc.add_cdlnks(self.lft_arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.lft_hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.lft_hnd.rgt, [1])
        self.cc.add_cdlnks(self.rgt_arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.rgt_hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.rgt_hnd.rgt, [1])
        activelist = [self.lft_arm.lnks[1],
                      self.lft_arm.lnks[2],
                      self.lft_arm.lnks[3],
                      self.lft_arm.lnks[4],
                      self.lft_arm.lnks[5],
                      self.lft_arm.lnks[6],
                      self.lft_hnd.lft.lnks[0],
                      self.lft_hnd.lft.lnks[1],
                      self.lft_hnd.rgt.lnks[1],
                      self.rgt_arm.lnks[1],
                      self.rgt_arm.lnks[2],
                      self.rgt_arm.lnks[3],
                      self.rgt_arm.lnks[4],
                      self.rgt_arm.lnks[5],
                      self.rgt_arm.lnks[6],
                      self.rgt_hnd.lft.lnks[0],
                      self.rgt_hnd.lft.lnks[1],
                      self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.lft_body.lnks[0],  # table
                    self.lft_body.lnks[1],  # body
                    self.lft_arm.lnks[1],
                    self.rgt_arm.lnks[1]]
        intolist = [self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.lft_arm.lnks[1],
                    self.lft_arm.lnks[2],
                    self.rgt_arm.lnks[1],
                    self.rgt_arm.lnks[2]]
        intolist = [self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.lft_arm.lnks[2],
                    self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.lft_hnd.lft.lnks[0],
                    self.lft_hnd.lft.lnks[1],
                    self.lft_hnd.rgt.lnks[1]]
        intolist = [self.rgt_arm.lnks[2],
                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self.rgt_hnd.lft.lnks[0],
                    self.rgt_hnd.lft.lnks[1],
                    self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)

    def get_hnd_on_manipulator(self, manipulator_name):
        if manipulator_name == 'rgt_arm':
            return self.rgt_hnd
        elif manipulator_name == 'lft_arm':
            return self.lft_hnd
        else:
            raise ValueError("The given jlc does not have a hand!")

    def fix_to(self, pos, rotmat):
        super().fix_to(pos, rotmat)
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                            rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        self.lft_hnd.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'],
                            rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        self.rgt_hnd.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: nparray 1x6 or 1x14 depending on component_names
        :hnd_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='rgt_arm'):
            # inline function for update objects in hand
            if component_name in ['rgt_arm', 'rgt_hnd']:
                oih_info_list = self.rgt_oih_infos
            elif component_name in ['lft_arm', 'lft_hnd']:
                oih_info_list = self.lft_oih_infos
            for obj_info in oih_info_list:
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

        super().fk(component_name, jnt_values)
        # examine length
        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        elif component_name == 'both_arm':
            if jnt_values.size != 14:
                raise ValueError("A 1x14 npdarrays must be specified to move both arm!")
            status_lft = update_component('lft_arm', jnt_values[0:7])
            status_rgt = update_component('rgt_arm', jnt_values[7:14])
            return "succ" if status_lft == "succ" and status_rgt == "succ" else "out_of_rng"
        elif component_name == 'all':
            raise NotImplementedError
        else:
            raise ValueError("The given component name is not available!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        elif component_name == 'both_arm':
            return_val = np.zeros(14)
            return_val[:7] = self.manipulator_dict['lft_arm'].get_jnt_values()
            return_val[7:] = self.manipulator_dict['rgt_arm'].get_jnt_values()
            return return_val
        else:
            raise ValueError("The given component name is not available!")

    def rand_conf(self, component_name):
        """
        override robot_interface.rand_conf
        :param component_name:
        :return:
        author: weiwei
        date: 20210406
        """
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        elif component_name == 'both_arm':
            return np.hstack((super().rand_conf('lft_arm'), super().rand_conf('rgt_arm')))
        else:
            raise NotImplementedError

    def hold(self, hnd_name, objcm, jaw_width=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :return:
        """
        if hnd_name == 'lft_hnd':
            rel_pos, rel_rotmat = self.lft_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.lft_body.lnks[0],
                        self.lft_body.lnks[1],
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self.rgt_hnd.lft.lnks[0],
                        self.rgt_hnd.lft.lnks[1],
                        self.rgt_hnd.rgt.lnks[1]]
            self.lft_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        elif hnd_name == 'rgt_hnd':
            rel_pos, rel_rotmat = self.rgt_arm.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
            intolist = [self.lft_body.lnks[0],
                        self.lft_body.lnks[1],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self.lft_hnd.lft.lnks[0],
                        self.lft_hnd.lft.lnks[1],
                        self.lft_hnd.rgt.lnks[1]]
            self.rgt_oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        return rel_pos, rel_rotmat

    def get_loc_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
        """
        get the loc pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name == 'lft_arm':
            arm = self.lft_arm
        elif component_name == 'rgt_arm':
            arm = self.rgt_arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return self.cvt_gl_to_loc_tcp(component_name, gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3])

    def get_gl_pose_from_hio(self, hio_pos, hio_rotmat, component_name='lft_arm'):
        """
        get the global pose of an object from a grasp pose described in an object's local frame
        :param hio_pos: a grasp pose described in an object's local frame -- pos
        :param hio_rotmat: a grasp pose described in an object's local frame -- rotmat
        :return:
        author: weiwei
        date: 20210302
        """
        if component_name == 'lft_arm':
            arm = self.lft_arm
        elif component_name == 'rgt_arm':
            arm = self.rgt_arm
        hnd_pos = arm.jnts[-1]['gl_posq']
        hnd_rotmat = arm.jnts[-1]['gl_rotmatq']
        hnd_homomat = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        hio_homomat = rm.homomat_from_posrot(hio_pos, hio_rotmat)
        oih_homomat = rm.homomat_inverse(hio_homomat)
        gl_obj_homomat = hnd_homomat.dot(oih_homomat)
        return gl_obj_homomat[:3, 3], gl_obj_homomat[:3, :3]

    def get_oih_cm_list(self, hnd_name='lft_hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        return_list = []
        for obj_info in oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def get_oih_glhomomat_list(self, hnd_name='lft_hnd'):
        """
        oih = object in hand list
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        return_list = []
        for obj_info in oih_infos:
            return_list.append(rm.homomat_from_posrot(obj_info['gl_pos']), obj_info['gl_rotmat'])
        return return_list

    def get_oih_relhomomat(self, objcm, hnd_name='lft_hnd'):
        """
        TODO: useless? 20210320
        oih = object in hand list
        :param objcm
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210302
        """
        if hnd_name == 'lft_hnd':
            oih_info_list = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_info_list = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        for obj_info in oih_info_list:
            if obj_info['collision_model'] is objcm:
                return rm.homomat_from_posrot(obj_info['rel_pos']), obj_info['rel_rotmat']

    def release(self, hnd_name, objcm, jaw_width=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jaw_width:
        :param objcm:
        :param hnd_name:
        :return:
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                oih_infos.remove(obj_info)
                break

    def release_all(self, jaw_width=None, hnd_name='lft_hnd'):
        """
        release all objects from the specified hand
        :param jaw_width:
        :param hnd_name:
        :return:
        author: weiwei
        date: 20210125
        """
        if hnd_name == 'lft_hnd':
            oih_infos = self.lft_oih_infos
        elif hnd_name == 'rgt_hnd':
            oih_infos = self.rgt_oih_infos
        else:
            raise ValueError("hnd_name must be lft_hnd or rgt_hnd!")
        if jaw_width is not None:
            self.jaw_to(hnd_name, jaw_width)
        for obj_info in oih_infos:
            self.cc.delete_cdobj(obj_info)
        oih_infos.clear()

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggle_tcpcs=toggle_tcpcs,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.lft_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggle_tcpcs=toggle_tcpcs,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel(toggle_tcpcs=False,
                                   toggle_jntscs=toggle_jntscs,
                                   rgba=rgba).attach_to(meshmodel)
        for obj_info in self.lft_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.attach_to(meshmodel)
        for obj_info in self.rgt_oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis

    base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    yumi_instance = Yumi(enable_cc=True)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    # yumi_instance.show_cdprimit()
    base.run()

    # ik test
    component_name = 'rgt_arm'
    tgt_pos = np.array([.4, -.4, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat)
    toc = time.time()
    print(toc - tic)
    yumi_instance.fk(component_name, jnt_values)
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    yumi_instance.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = yumi_instance.is_collided()
    toc = time.time()
    print(result, toc - tic)

    # hold test
    component_name = 'lft_arm'
    obj_pos = np.array([-.1, .3, .3])
    obj_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    objfile = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm = cm.CollisionModel(objfile, cdprimit_type='cylinder')
    objcm.set_pos(obj_pos)
    objcm.set_rotmat(obj_rotmat)
    objcm.attach_to(base)
    objcm_copy = objcm.copy()
    yumi_instance.hold(objcm=objcm_copy, jaw_width=0.03, hnd_name='lft_hnd')
    tgt_pos = np.array([.4, .5, .4])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
    jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat)
    yumi_instance.fk(component_name, jnt_values)
    yumi_instance.show_cdprimit()
    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)

    base.run()
