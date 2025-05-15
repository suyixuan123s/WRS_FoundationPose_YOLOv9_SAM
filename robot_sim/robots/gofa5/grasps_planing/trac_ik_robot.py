# 生成無碰撞点合集
# 修改x,y,z的间隔可以调整精度
import math

import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import robot_sim.robots.gofa5.gofa5 as gofa5
import basis.robot_math as rm

# 生成窗口
base = wd.World(cam_pos=[10, -5, 2], lookat_pos=[3, 0, 0])
gm.gen_frame().attach_to(base)
# 生成初始机械臂
robot_s = gofa5.GOFA5(enable_cc=True)
component_name = 'arm'

pos = [.6, .1, .1]
rotmat = rm.rotmat_from_euler(0, math.radians(90), 0)
rotmat = np.eye(3)
gm.gen_frame(pos=pos, rotmat=rotmat, length=1).attach_to(base)

seed_jnt_values = np.array([0, 0, 0, 0, 0, 0])
jnt_values = robot_s.ik(
    component_name="arm",
    tgt_pos=pos,
    tgt_rotmat=rotmat,
    seed_jnt_values=np.array([0, 0, 0, 0, 0, 0]),
    max_niter=1000,
)

print(jnt_values)

if jnt_values is not None:
    robot_s.fk("arm", jnt_values)
    robot_s.gen_meshmodel(rgba=[1, 0, 0, 1]).attach_to(base)

base.run()

#
# #设置所有刀的点位
# ymax=0
# zmax=0
# ymin=0
# zmin=0
# # 初始化空的 NumPy 数组
# dot_array = np.empty((0,), dtype=object)  # 使用 object 类型,以支持任意数据
# pose_array = np.empty((0,), dtype=object)
# i=0
# tgt_rotmat = np.eye(3)
# target_center = np.array([5.25, -1.081, -2.279])
#
#
# rot_x_90 = np.array([
#                     [1, 0, 0],
#                     [0, 0, -1],
#                     [0, 1, 0]
#                 ])
# rot_y_90 = np.array([
#                         [0, 0, 1],
#                         [0, 1, 0],
#                         [-1, 0, 0]
#                             ])
# rot_z_180 = np.array([
#                         [-1, 0, 0],
#                         [0, -1, 0],
#                         [0, 0, 1]
#                             ])
#
# for x in np.arange(4.7,5.8,1):
#     x = round(x, 3)
#     print('x',x)
#     for z in np.arange(-0.1, 2, 1):
#         z=round(z, 3)
#         for y in np.arange(-1.3, 1,1):
#             y = round(y, 3)
#             tgt_pos = np.array([x, y, z])
#             theta = np.arctan2(target_center[2] - z,target_center[1] - y  )
#
#             # 例如 45 度的旋转
#             # theta = np.pi / 3
#             tgt_rotmat = np.array([
#                 [np.cos(theta), -np.sin(theta), 0],
#                 [np.sin(theta), np.cos(theta), 0],
#                 [0, 0, 1]
#             ])
#             origin_rotmat = np.dot(rot_y_90,rot_z_180)
#             tgt_rotmat = np.dot(origin_rotmat,tgt_rotmat)
#             # tgt_rotmat =rot_y_90
#
#             component_name = 'arm'
#
#             pose = robot_s.ik(
#                                 component_name= "arm",
#                                 tgt_pos=tgt_pos,
#                                 tgt_rotmat=tgt_rotmat,
#                                 seed_jnt_values=np.array([2,0,0,0,0,0,0]),
#                                 max_niter=1000,
#             )
#
#             if pose is not None:
#                 # for pose in poses:
#                 print(pose)
#
#                 robot_s.fk(component_name='arm',jnt_values=pose)
#                 flag_collide = robot_s.is_collided(obstacle_list=obstacle_list)
#                 if flag_collide is False:
#                     robot_s.fk(component_name='arm', jnt_values=pose)
#                     robot_s.gen_meshmodel().attach_to(base)
#
#                     dot = tgt_pos
#                     # 使用 np.vstack(垂直拼接)来动态增加数组
#                     dot_array = np.vstack([dot_array, dot]) if dot_array.size else np.array([dot])
#                     pose_array = np.vstack([pose_array, pose]) if pose_array.size else np.array([pose])
#                     break
#
#
#
# # data = dot_list
# # with open('map_dot30.csv', 'w', newline='') as csvfile:
# #     writer = csv.writer(csvfile)
# #     for row in data:
# #         writer.writerow(row)
# # data = pose_list
# # with open('map_pose30.csv', 'w', newline='') as csvfile:
# #     writer = csv.writer(csvfile)
# #     for row in data:
# #         writer.writerow(row)
# # 写入 CSV 文件
# np.savetxt('map_dot_cdmesh_box_30_311.csv', dot_array, delimiter=",", fmt='%s')  # fmt='%s' 以确保支持字符串和数值
# np.savetxt('map_pose_cdmesh_box_30_311.csv', pose_array, delimiter=",", fmt='%s')
# # for x in np.arange(4.7,5.8,0.03):
# #     print(x)
# #     for z in np.arange(-0.1, 2, 0.03):
# #         for y in np.arange(-1.3, 1, 0.03):
# object_box1 = cm.gen_box(extent=[1.1, 2.3, 2.1])
# pos1 = [5.25, -0.15, 0.875]
# object_box1.set_pos(np.array(pos1))
# object_box1.set_rgba([.5, .7, .5, 0.5])
# object_box1.attach_to(base)
#
# #大刀盘
# daopan = cm.CollisionModel("./objects/daopan_zeroyuanxin.stl")
# daopan.change_cdprimitive_type(cdprimitive_type='polygons')
# # daopan.change_cdmesh_type(cdmesh_type='triangles')
# # object.set_pos(np.array([2.835+0.52+2, -0.92-0.2, -3.1+0.67]))
# # 0.52, -0.2, .67
# # pos = np.array([5.02+0.5, -1.081, -2.279-0.05-0.3])
# # pos = np.array([5.02-0.1, -1.081-1.1, -2.279])
# # pos = np.array([5.02-0.8, -1.081, -2.279])
# pos = np.array([5.39-0.2,-1.081,-2.279])
# rotmat = rm.rotmat_from_euler(math.radians(20), 0, 0)
# daopan.set_pos(pos)
# daopan.set_rotmat(rotmat)
# # self.base_plate.lnks[0]['loc_pos'] = np.array([2.835, -0.92, -3.1])
# daopan.set_rgba([.5,.7,.5,1])
# daopan.attach_to(base)
#
# # 在循环结束后统一渲染
# for dot in dot_array:
#     gm.gen_sphere(pos=dot, radius=0.015,subdivisions=0).attach_to(base)
# base.run()
