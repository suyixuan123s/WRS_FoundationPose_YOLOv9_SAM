"""
Author: Yixuan Su
Date: 2025/04/28 22:46
File: grasping_antipodal_planning_ag145_blood_tube10.py
Description:

"""

import os
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.ag145.ag145 as ag145
import basis.robot_math as rm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
this_dir, this_filename = os.path.split(__file__)
objpath = os.path.join(this_dir, 'objects', 'blood_tube10.STL')
object_tube = cm.CollisionModel(objpath)
object_tube.set_rgba([1, 0, 0, 1])
object_tube.attach_to(base)
gripper_s = ag145.Ag145()

jaw_width = 0.02
jaw_center_pos = np.array([0, 0, 0.0185 / 2])
axis = [0, 1, 0]
angle = np.pi
rotmat_y = rm.rotmat_from_axangle(axis, angle)
axis1 = [0, 0, 1]

# 初始化抓取信息列表
grasp_info_list = []
# 逐步旋转并保存抓取信息
for angle_deg in range(0, 360, 30):  # 从0度到359度,每次增加60度
    angle_rad = np.radians(angle_deg)  # 将角度转换为弧度
    rotmat_z = rm.rotmat_from_axangle(axis1, angle_rad)  # 计算旋转矩阵
    # jaw_center_rotmat: 先绕Y轴旋转,再绕Z轴旋转
    jaw_center_rotmat = np.dot(rotmat_z, rotmat_y)
    # 获取抓取信息列表
    grasp_info = gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    grasp_info_list.append(grasp_info)

gpa.write_pickle_file('blood_tube10', grasp_info_list, './', './ag145_grasps_blood1.pickle')
# print("抓取信息列表:", grasp_info_list)

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    print("grasp_info", grasp_info)
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    print(jaw_width)

grasp_info_list = gpa.load_pickle_file('blood_tube10', root=None, file_name='./ag145_grasps_blood1.pickle')
print("grasp_info_list", grasp_info_list)
base.run()
