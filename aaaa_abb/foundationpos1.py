import numpy as np
from direct.task.TaskManagerGlobal import taskMgr
from numpy.linalg import inv
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
from panda3d.core import NodePath
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta as cbt
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc


if __name__ == '__main__':
    # 1. 创建 Panda3D 世界
    base = wd.World(cam_pos=[-0.5 ,0.4, -0.8], lookat_pos=[0, 0, 0])

    table = cm.gen_box([2, 2, 0.001])
    table.set_rgba((0.51, 0.51, 0.51, 1))
    table.set_pos((0, 0, 0.095))
    table.attach_to(base)


    # 2. 定义物体相对于相机的转换矩阵
    T_camera_object = np.array([[1, -0, 0, 0.002693533897],
                                [-0, 1, -0, -0.008149208501],
                                [0, -0, 1, 0.6390047073],
                                [-0, 0, -0, 1]])

    # 3. 计算相机相对于物体的转换矩阵
    T_object_camera = inv(T_camera_object)

    # 4. 提取相机在世界坐标系中的位置 (由于物体在世界坐标系原点)
    camera_position_in_world_frame = T_object_camera[:3, 3]
    print("相机在世界坐标系中的位置:", camera_position_in_world_frame)

    # 5. 创建物体 (在世界坐标系原点)
    object_model = cm.CollisionModel("textured_mesh.obj")
    object_model.set_rgba([0, 1, 0, 0.5])
    object_model.attach_to(base)


    # 6. 创建相机 (StaticGeometricModel)
    camera_model = gm.gen_box([0.05, 0.01, 0.02])
    camera_model.set_rgba([0, 0, 1, 1])  # 蓝色

    # 7. 创建 NodePath 对象
    camera_node = NodePath("camera_node")

    # 8. 将几何模型附加到 NodePath 对象
    camera_model.attach_to(camera_node)

    # 9. 将 NodePath 对象放置在世界坐标系中的计算位置
    camera_node.setPos(camera_position_in_world_frame[0],
                       camera_position_in_world_frame[1],
                       camera_position_in_world_frame[2])

    # 10. 将 NodePath 对象附加到 base
    camera_node.reparentTo(base.render)

    # # 11. 显示坐标轴
    gm.gen_frame().attach_to(base)

    def compute_position_and_orientation(T_transform):
        """
        直接计算物体在相机坐标系中的位置和姿态
        :param T_transform: 物体到相机坐标系的转换矩阵 (4x4)
        :return: 物体在相机坐标系中的位置和姿态
        """
        # 提取物体在相机坐标系中的位置(平移向量)
        position_camera = T_transform[:3, 3]

        # 提取物体在相机坐标系中的旋转(姿态)矩阵
        rotation_matrix_camera = T_transform[:3, :3]

        return position_camera, rotation_matrix_camera


    # 物体到相机坐标系的转换矩阵
    T_transform = np.array([
        [0.9986738563, -0.001494958298, -0.05146124214, 0.002847550204],
        [0.004152976442, 0.9986599684, 0.0515823476, -0.008390889503],
        [0.05131517351, -0.05172766373, 0.9973418713, 0.6413608193],
        [0, 0, 0, 1]
    ])

    # 计算物体在相机坐标系中的位置和姿态
    position_camera, rotation_matrix_camera = compute_position_and_orientation(T_transform)

    # 输出结果
    print("物体在相机坐标系中的位置: ", position_camera)
    print("物体在相机坐标系中的姿态: \n", rotation_matrix_camera)


    def transform_pose(T_initial, T_transform):
        """
        计算物体经过位姿转换后的位姿
        :param T_initial: 初始位姿矩阵 (4x4 齐次变换矩阵)
        :param T_transform: 位姿转换矩阵 (4x4 齐次变换矩阵)
        :return: 经过转换后的新位姿矩阵
        """
        # 矩阵相乘得到新位姿
        T_new = np.dot(T_transform, T_initial)
        return T_new


    # 示例初始位姿和位姿转换矩阵
    # 初始位姿矩阵 (4x4)
    T_initial = np.array([
        [1, 0, 0, 0.002693533897],
        [0, 1, 0, -0.008149208501],
        [0, 0, 1, 0.6390047073],
        [0, 0, 0, 1]
    ])

    # 转换后的位姿矩阵 (4x4)
    T_transform = np.array([
        [0.9986738563, -0.001494958298, -0.05146124214, 0.002847550204],
        [0.004152976442, 0.9986599684, 0.0515823476, -0.008390889503],
        [0.05131517351, -0.05172766373, 0.9973418713, 0.6413608193],
        [0, 0, 0, 1]
    ])

    # 计算经过位姿转换后的新位姿
    T_new = transform_pose(T_initial, T_transform)

    print("经过转换后的新位姿矩阵: ")
    print(T_new)

    # 位置插值
    num_steps = 300
    pos_initial = T_initial[:3, 3]
    pos_final = T_transform[:3, 3]
    pos_list = np.linspace(pos_initial, pos_final, num=num_steps)

    # 旋转插值(使用矩阵直接插值)
    rot_initial = T_initial[:3, :3]
    rot_final = T_transform[:3, :3]

    # 创建旋转插值的矩阵
    rot_list = []
    for t in np.linspace(0, 1, num=num_steps):
        rot_interpolated = (1 - t) * rot_initial + t * rot_final
        rot_list.append(rot_interpolated)

    # 组合插值结果,生成新的位姿矩阵
    box_conf_list = []
    for i in range(num_steps):
        T_interpolated = np.eye(4)
        T_interpolated[:3, 3] = pos_list[i]  # 插值位置
        T_interpolated[:3, :3] = rot_list[i]  # 插值旋转
        box_conf_list.append(T_interpolated)


    pos_list = np.linspace([0, 0, 0.01], [1, 0, 0.01], 300)
    rot_list = [np.eye(3) for item in pos_list]
    box_conf_list = [rm.homomat_from_posrot(pos_list[i], rot_list[i]) for i in range(len(pos_list))]

    counter = [0]
    flag = [0]


    def update(box_conf_list, counter, flag, task):
        object_model.set_homomat(box_conf_list[counter[0]])
        object_model.attach_to(base)
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


