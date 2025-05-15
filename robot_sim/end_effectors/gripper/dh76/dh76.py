import os
import math
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
from typing import Literal
import robot_sim.end_effectors.gripper.gripper_interface as gp



class Dh76(gp.GripperInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3), cdmesh_type='box', name='Dh60',
                 fingertip_type: Literal['l_76', 'r_76'] = 'l_76', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # self.coupling.jnts[1]['loc_pos'] = coupling_offset_pos
        # self.coupling.jnts[1]['loc_rotmat'] = coupling_offset_rotmat
        # self.coupling.lnks[0]['collision_model'] = cm.gen_stick(self.coupling.jnts[0]['loc_pos'],
        #                                                         self.coupling.jnts[1]['loc_pos'],
        #                                                         thickness=.07, rgba=[.2, .2, .2, 1],
        #                                                         sections=24)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']

        # fingertip
        self.fingertip_type = fingertip_type
        if self.fingertip_type == 'l_76':
            # 切换成加长版手指,方接触面,76行程
            self.fingertip_1 = os.path.join(this_dir, "meshes", "long_fingertip_76_1.stl")
            self.fingertip_2 = os.path.join(this_dir, "meshes", "long_fingertip_76_2.stl")

        elif self.fingertip_type == 'r_76':
            # 切换成加长版手指,圆接触面,76行程
            self.fingertip_1 = os.path.join(this_dir, "meshes", "round_long_fingertip_76_1.stl")
            self.fingertip_2 = os.path.join(this_dir, "meshes", "round_long_fingertip_76_2.stl")

        # lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='base_lft_finger')
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        self.lft.lnks[0]['rgba'] = [.2, .2, .2, 1]

        self.lft.jnts[1]['loc_pos'] = np.array([-.0018, -.01635, .14])
        self.lft.jnts[1]['type'] = 'prismatic'
        self.lft.jnts[1]['motion_rng'] = [0, .038]
        self.lft.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['mesh_file'] = cm.CollisionModel(
            self.fingertip_1, cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._lftfinger_cdnp, expand_radius=.000)
        self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]
        self.lft.jnts[2]['loc_pos'] = np.array([.0294, .01635, -.007])
        self.lft.jnts[2]['loc_rotmat'] = np.array([0, 0, 0])
        self.lft.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 0)

        # rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([.0018, .01635, .14])
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['mesh_file'] = cm.CollisionModel(
            self.fingertip_2, cdprimit_type="user_defined",
            userdefined_cdprimitive_fn=self._rgtfinger_cdnp, expand_radius=.000)
        self.rgt.lnks[1]['rgba'] = [.5, .5, .5, 1]
        self.rgt.jnts[2]['loc_pos'] = np.array([-.0294, -.01635, -.007])
        self.rgt.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)

        # jaw center
        self.jaw_center_pos = np.array([0, 0, .2035]) + coupling_offset_pos

        # jaw width
        self.jawwidth_rng = [.0, .076]

        # reinitialize
        self.lft.reinitialize()
        self.rgt.reinitialize()

        # collision detection
        self.all_cdelements = []
        self.enable_cc(toggle_cdprimit=enable_cc)

    @staticmethod
    def _lftfinger_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.0104, .01635, .034),
                                              x=.01 + radius, y=0.012 + radius, z=.043 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    @staticmethod
    def _rgtfinger_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.0104, -.01635, .034),
                                              x=.01 + radius, y=0.012 + radius, z=.043 + radius)
        collision_node.addSolid(collision_primitive_c0)
        return collision_node

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft, [0, 1])
            self.cc.add_cdlnks(self.rgt, [1])
            activelist = [self.lft.lnks[0],
                          self.lft.lnks[1],
                          self.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements

        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, meter or radian
        """
        if self.lft.jnts[1]['motion_rng'][0] <= motion_val <= self.lft.jnts[1]['motion_rng'][1]:
            self.lft.jnts[1]['motion_val'] = motion_val
            self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=jaw_width / 2.0)

    def get_jawwidth(self):
        return self.lft.jnts[1]['motion_val'] * 2

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='lite6wrs_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(stickmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xc330gripper'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(meshmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel

    def open(self):
        '''
        gripper open
        '''
        self.jaw_to(.076)

    def close(self):
        '''
        gripper close
        '''
        self.jaw_to(0)


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    gm.gen_frame().attach_to(base)
    # cm.CollisionModel("meshes/dual_realsense.stl", expand_radius=.001).attach_to(base)
    grpr = Dh76(enable_cc=True)
    grpr.open()
    grpr.show_cdprimit()
    grpr.gen_meshmodel().attach_to(base)
    base.run()
