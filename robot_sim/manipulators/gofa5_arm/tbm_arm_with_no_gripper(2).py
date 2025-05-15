import os
import math
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi
import modeling.geometric_model as gm
import modeling.model_collection as mc
from constants import TBM_ARM_ORIGIN2BASE, np

try:
    from trac_ik import TracIK

    is_trac_ik = True
    print("Track IK starts")
except Exception as e:
    print("Track IK not available")
    print(e)
    is_trac_ik = False


class TBMArm(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='tbm', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # cdprimitive的种类['box',  'surface_balls',  'cylinder','polygons', 'point_cloud', 'user_defined']
        # cdmesh的种类['aabb',  'obb', 'convex_hull', 'triangles']

        self.jlc = jl.JLChain(pos=pos,
                              rotmat=rotmat,
                              homeconf=homeconf,
                              cdprimitive_type='box',
                              cdmesh_type='obb',
                              name=name)
        # 7 joints, n_jnts = 7+2, n_links = 7+1

        # 1 base slide
        # self.jlc.jnts[1]['loc_pos'] = np.array([.555-x,- 0.13-x, 0.15+x])
        # self.jlc.jnts[1]['loc_pos'] = np.array([0.555, -0.1957, 0.67])
        self.jlc.jnts[1]['loc_pos'] = np.array([TBM_ARM_ORIGIN2BASE, 0, 0])  # 455
        # self.jlc.jnts[1]['loc_pos'] = np.array([0, -0, 0])
        self.jlc.jnts[1]['type'] = 'prismatic'
        self.jlc.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(-1.5707963267948966, 0., -1.5707963267948966)
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[1]['motion_rng'] = [0, 3.010]

        self.jlc.jnts[2]['loc_pos'] = np.array([0, -0.2225 - 0.035, 0])
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(1.5707963267948966, 0, 0)
        self.jlc.jnts[2]['motion_rng'] = [-math.radians(60), math.radians(60)]

        self.jlc.jnts[3]['loc_pos'] = np.array([0, 0.572, .0])  # 580mm -》 572mm (-8mm)
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-1.5707963267948966, -1.5707963267948966, .0)
        self.jlc.jnts[3]['motion_rng'] = [-math.radians(180), math.radians(180)]

        self.jlc.jnts[4]['loc_pos'] = np.array([0, .0, .430])  # 405mm->430mm (+25mm)
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(0, 1.5707963267948966, 0.0)
        self.jlc.jnts[4]['motion_rng'] = [-math.radians(90), math.radians(90)]

        self.jlc.jnts[5]['loc_pos'] = np.array([-0.5495, .0, .0])  # 556.5 mm -> 549.5mm (-7mm)
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(1.5707963267948966, 0, -1.5707963267948966)
        self.jlc.jnts[5]['motion_rng'] = [-math.radians(180), math.radians(180)]

        self.jlc.jnts[6]['loc_pos'] = np.array([0, .0, 0.5105])  # 559mm-> 510.5mm (-48.5mm)
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(0, -1.5707963267948966, -1.5707963267948966)
        self.jlc.jnts[6]['motion_rng'] = [-math.radians(100), math.radians(100)]

        self.jlc.jnts[7]['loc_pos'] = np.array([.2190, .0, .0])
        self.jlc.jnts[7]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-1.5707963267948966, 0, -1.5707963267948966)
        self.jlc.jnts[7]['motion_rng'] = [-math.radians(360), math.radians(360)]

        self.jlc.lnks[1]['name'] = "base"
        self.jlc.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.lnks[1]['loc_rotmat'] = rm.rotmat_from_euler(0.7853981633974483, -1.5707963267948966,
                                                              0.7853981633974483)
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrsbase.stl")
        self.jlc.lnks[1]['rgba'] = [.5, .5, .5, 1]

        self.jlc.lnks[2]['name'] = "j1"
        self.jlc.lnks[2]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[2]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, 1.5707963267948966)
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink1.stl")
        self.jlc.lnks[2]['rgba'] = [.7, .7, .7, 1]

        self.jlc.lnks[3]['name'] = "j2"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink2.stl")
        self.jlc.lnks[3]['rgba'] = [.77, .77, .60, 1]
        # self.jlc.lnks[3]['loc_rotmat'] = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)

        self.jlc.lnks[4]['name'] = "j3"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .007])
        self.jlc.lnks[4]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, np.pi)
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink3.stl")
        self.jlc.lnks[4]['rgba'] = [.35, .35, .35, 1]

        self.jlc.lnks[5]['name'] = "j4"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, -.0485])
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink4.stl")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]
        # self.jlc.lnks[5]['loc_rotmat'] = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)

        self.jlc.lnks[6]['name'] = "j5"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, 0])
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "tbmwrslink5.stl")
        self.jlc.lnks[6]['rgba'] = [.77, .77, .60, 1]

        self.jlc.lnks[7]['name'] = "j6"
        self.jlc.lnks[7]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[7]['mesh_file'] = None
        self.jlc.lnks[7]['rgba'] = [.5, .5, .5, 1]

        self.jlc.reinitialize()
        # collision detection
        if enable_cc:
            self.enable_cc()

        # cd meshes collection for precise collision checking
        self.cdmesh_collection = mc.ModelCollection()
        if is_trac_ik:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf = os.path.join(directory, "tbm_arm.urdf")
            self._ik_solver = TracIK("world", f"{self.name}_link_7", urdf)
            self._ik_solver_6dof = TracIK(f"{self.name}_link_1", f"{self.name}_link_7", urdf)
            self.offset_homomat = np.linalg.inv(rm.homomat_from_posrot(self.jlc.jnts[1]['gl_pos0'],
                                                                       self.jlc.jnts[1][
                                                                           'gl_rotmat0'])).dot(
                rm.homomat_from_posrot(self.jlc.jnts[2]['gl_pos0'],
                                       self.jlc.jnts[2]['gl_rotmat0']))
        else:
            self._ik_solver = None
            self._ik_solver_6dof = None

    def get_gl_tcp_coord0(self,
                          tcp_jnt_id=None,
                          tcp_loc_pos=None,
                          tcp_loc_rotmat=None):
        pos, rot = self.jlc.get_gl_tcp(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat)
        id0 = 2
        pos0, rot0 = self.jlc.jnts[id0]['gl_pos0'], self.jlc.jnts[id0]['gl_rotmat0']
        matbase2tcp = np.linalg.inv(rm.homomat_from_posrot(pos0, rot0)).dot(rm.homomat_from_posrot(pos, rot))
        return matbase2tcp[:3, 3], matbase2tcp[:3, :3]

    def fk6(self, jnt_values):
        j = self.get_jnt_values()
        j[1:] = jnt_values
        return self.jlc.fk(jnt_values=j)

    def ik(self,
           tgt_pos=np.array([.7, 0, .7]),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=500,
           tcp_jntid=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima="accept",
           toggle_debug=False):
        if self._ik_solver is None:
            return super().ik(tgt_pos, tgt_rotmat, seed_jnt_values, max_niter, tcp_jntid,
                              tcp_loc_pos, tcp_loc_rotmat, local_minima, toggle_debug)
        else:
            seed_jnt_values = self.homeconf if seed_jnt_values is None else seed_jnt_values.copy()
            ik_solution = self._ik_solver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values)
            # print("ik solution is", ik_solution, tgt_pos, tgt_rotmat)
            return ik_solution

    def ik_6dof(self,
                tgt_pos=np.array([.7, 0, .7]),
                tgt_rotmat=np.eye(3),
                seed_jnt_values=None, ):
        if self._ik_solver_6dof is None:
            raise NotImplementedError
        else:
            tgt_homomat = self.offset_homomat.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]
            seed_jnt_values = self.homeconf if seed_jnt_values is None else seed_jnt_values.copy()
            ik_solution = self._ik_solver_6dof.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values[1:7])
            return ik_solution

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[1],
                    self.jlc.lnks[2],
                    self.jlc.lnks[3]]
        intolist = [self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)

    def set_tracik_joint_limits(self, lower_bounds, upper_bounds):
        if is_trac_ik:
            self._ik_solver.joint_limits = [lower_bounds, upper_bounds]
            self._ik_solver_6dof.joint_limits = [lower_bounds[1:], upper_bounds[1:]]
        else:
            raise Exception("Trac IK is not Correctly installed")

    def get_tracik_joint_limits(self):
        if is_trac_ik:
            return self._ik_solver.joint_limits
        else:
            raise Exception("Trac IK is not Correctly installed")


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3.7, -4, 1.7], lookat_pos=[1.5, 0, .3])
    gm.gen_frame().attach_to(base)
    manipulator_instance = TBMArm(enable_cc=True)
    lm = np.asarray(manipulator_instance.get_jnt_ranges())
    lb = lm[:,0]
    ub = lm[:, 1]
    print(lm, lb, ub)

    print("length:", manipulator_instance.get_gl_tcp_coord0()[0])
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    for cm in manipulator_meshmodel.cm_list:
        cm.set_rgba([*cm.get_rgba()[:3], .3])
        cm.attach_to(base)

    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    manipulator_instance.show_cdprimit()
    with open("tbm_arm.urdf", "w") as f:
        f.write(str(manipulator_instance.gen_urdf()))
    base.run()
