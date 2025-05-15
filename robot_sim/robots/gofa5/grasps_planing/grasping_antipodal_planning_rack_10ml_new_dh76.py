# 这段代码实现了一个基于仿真环境的抓取规划和控制任务
# 通过加载物体模型、计算抓取点并控制夹爪进行抓取,展示了机器人在仿真中的抓取过程
# 同时,计算出的抓取信息被保存到文件中,方便后续使用

import math
import os
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.ag145.ag145 as ag145

import robot_sim.end_effectors.gripper.dh76.dh76 as dh76


base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
this_dir, this_filename = os.path.split(__file__)
objpath = os.path.join(this_dir, 'objects', 'rack_10ml_new.stl')
object_tube = cm.CollisionModel(objpath)
object_tube.set_rgba([1, 0, 0, 1])
object_tube.attach_to(base)
# gripper_s = ag145.Ag145()

gripper_s = dh76.Dh76(fingertip_type='r_76')

# 根据物体模型和夹爪类型来规划抓取点
grasp_info_list = gpa.plan_grasps(gripper_s,
                                  object_tube,
                                  angle_between_contact_normals=math.radians(177),
                                  openning_direction='loc_x',
                                  max_samples=100,
                                  min_dist_between_sampled_contact_points=.003,
                                  contact_offset=.003)
gpa.write_pickle_file('rack_10ml_new', grasp_info_list, './', 'dh76_grasps_rack_10ml_new_r_7611.pickle')

for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    print(jaw_width)
base.run()
