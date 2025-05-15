from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm

from scipy.spatial.transform import Rotation as R, Slerp

if __name__ == '__main__':
    base = wd.World(cam_pos=[-0.00269353, 0.00814921, -0.63900471], lookat_pos=[0, 0, .9])
    gm.gen_frame().attach_to(base)
    table = cm.gen_box([2, 1, 0.001])
    table.set_rgba((0.51, 0.51, 0.51, 1))
    table.set_pos((0, 0, 0.0005))

    object_tube = cm.CollisionModel("textured_mesh.obj")
    object_tube.set_pos()
    object_tube.set_rgba((0, 1, 0, 0.5))
    table.attach_to(base)
    object_tube.attach_to(base)
    # base.run()

T_initial = np.array([
    [1, 0, 0, 0.002693533897],
    [0, 1, 0, -0.008149208501],
    [0, 0, 1, 0.6390047073],
    [0, 0, 0, 1]
])

T_transform = np.array([
    [0.3547237813, -0.9132992625, 0.2001387477, 0.02930317074],
    [-0.09771322459, 0.1766738147, 0.9794072509, -0.02930317074],
    [-0.9298511147, -0.3669753075, -0.02657106891, 0.02930317074],
    [0, 0, 0, 1]
])

translation_start = T_initial[:3, 3]
translation_end = T_transform[:3, 3]
rotation_start = T_initial[:3, :3]
rotation_end = T_transform[:3, :3]

# 位置插值
pos_list = np.linspace(translation_start, translation_end, 300)

# 旋转插值(使用Slerp)
# 修正后的旋转插值部分

# 将旋转矩阵转换为Rotation对象
rotation_start = R.from_matrix(T_initial[:3, :3])
rotation_end = R.from_matrix(T_transform[:3, :3])

# 创建时间序列和旋转序列
times = [0, 1]
key_rots = R.from_matrix([T_initial[:3, :3], T_transform[:3, :3]])

# 创建Slerp实例
slerp = Slerp(times, key_rots)

# 生成插值后的旋转矩阵
rot_list = slerp(np.linspace(0, 1, 300)).as_matrix()  # 转换为numpy矩阵

# 生成齐次矩阵列表
box_conf_list = [rm.homomat_from_posrot(pos_list[i], rot_list[i])
                 for i in range(len(pos_list))]

counter = [0]
flag = [0]


def update(box_conf_list, counter, flag, task):
    object_tube.set_homomat(box_conf_list[counter[0]])
    object_tube.attach_to(base)
    if flag[0] == 0:
        counter[0] += 1
        if counter[0] == len(box_conf_list) - 1:
            flag[0] = 1
    else:
        counter[0] = counter[0] - 1
        if counter[0] == 0:
            flag[0] = 0

    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[box_conf_list, counter, flag],
                      appendTask=True)

base.run()
