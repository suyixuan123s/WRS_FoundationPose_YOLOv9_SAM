import robot_sim.end_effectors.gripper.ag145.ag145 as ag

gripper_s = ag.Ag145()
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.gofa5.gofa5_dh76 as gf5
import modeling.collision_model as cm
import basis.robot_math as rm
import math
import numpy as np

base = wd.World(cam_pos=[2.554, 2.5, 2.5], lookat_pos=[0, 0, 0])  ##看向的位置
gm.gen_frame().attach_to(base)
tube = cm.CollisionModel(r"E:\ABB-Project\ABB_wrs_hu\robot_sim\robots\gofa5\grasps_planing\objects\blood_tube10.STL")
tube.attach_to(base)
print(tube.get_com())
# com=gm.gen_sphere(tube.get_com(),radius=0.01)
# com.attach_to(base)
cup_pos = [0.0101, 0.0101, 0.098]
gm.gen_sphere(cup_pos, radius=0.01).attach_to(base)
cup_rotmat = tube.get_rotmat()
gm.gen_frame(cup_pos, cup_rotmat, length=0.15).attach_to(base)
jaw_rotmat = rm.rotmat_from_axangle(axis=[0, 1, 0], angle=math.pi)
print(jaw_rotmat)
gripper_s.grip_at_with_jcpose(cup_pos, jaw_rotmat, 0.02)
gripper_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
z_90 = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=math.pi / 2)
rotmat1 = np.dot(z_90, jaw_rotmat)
gripper_s.grip_at_with_jcpose(cup_pos, rotmat1, 0.02)
gripper_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 1, 0, 0.1]).attach_to(base)
rotmat2 = np.dot(z_90, rotmat1)
gripper_s.grip_at_with_jcpose(cup_pos, rotmat2, 0.02)
gripper_s.gen_meshmodel(toggle_tcpcs=True, rgba=[1, 0, 0, 0.1]).attach_to(base)
rotmat3 = rotmat2 = np.dot(z_90, rotmat2)
gripper_s.grip_at_with_jcpose(cup_pos, rotmat3, 0.02)
gripper_s.gen_meshmodel(toggle_tcpcs=True, rgba=[0, 0, 1, 0.1]).attach_to(base)
base.run()
