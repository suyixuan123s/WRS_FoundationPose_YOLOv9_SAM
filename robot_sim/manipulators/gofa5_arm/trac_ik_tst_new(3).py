""" 
Author: Hao Chen (chen960216@gmail.com)
Created: 20220811osaka
"""

import visualization.panda.world as wd
import modeling.geometric_model as gm
from trac_ik import TracIK
import os
import numpy as np
import robot_sim.manipulators.gofa5_arm.gofa5_arm as gofa5_arm


base = wd.World(cam_pos=[7, 8, 3], lookat_pos=[4, 0, 1])
gm.gen_frame().attach_to(base)
manipulator_instance = gofa5_arm.Gofa5Arm(enable_cc=True)
val = manipulator_instance.rand_conf()
print(val)
# base.run()

manipulator_instance.fk(val)
manipulator_instance.gen_meshmodel().attach_to(base)
# print(manipulator_instance.get_gl_tcp())
directory = os.path.abspath(os.path.dirname(__file__))

# # 获取上两级目录的绝对路径,得到wrs_tbm所在位置
# grandparent_directory = os.path.abspath(os.path.join(directory, "..", ".."))
# # 双引号后面加入机械臂的文件夹位置
# directory = os.path.abspath(os.path.join(grandparent_directory, "manipulators/gofa5_arm"))
# # 双引号后面加入机械臂的文件名
yumi_urdf = os.path.join(directory, "tbm_arm.urdf")

# ik_solver = TracIK("world", "tbm_link_7", yumi_urdf)
ik_solver = TracIK("world", "gofa5_arm_link_6", yumi_urdf, solver_type="Manip2")


pos = [.5, .1, .1]
rotmat = np.eye(3)
seed_jnt_values = np.array([0, 0, 0, 0, 0, 0])
pose = ik_solver.ik(pos, rotmat, seed_jnt_values)
print(pose)
# [ 2.91828751  0.43456978  1.16943558  0.56628865  0.46596667  1.05426391
#  -0.67801117]
manipulator_instance.fk(pose)
manipulator_instance.gen_meshmodel(rgba=[1, 0, 0, 1]).attach_to(base)

base.run()


