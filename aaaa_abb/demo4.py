from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.robot_math as rm
import numpy as np

if __name__ == '__main__':
    base = wd.World(cam_pos=[-0.00269353, 0.00814921, -0.63900471], lookat_pos=[0, 0, .9])
    gm.gen_frame().attach_to(base)
    table = cm.gen_box([2, 1, 0.001])
    table.set_rgba((0.51, 0.51, 0.51, 1))
    table.set_pos((0, 0, 0.0005))
    table.attach_to(base)

    object_tube = cm.CollisionModel("textured_mesh.obj")
    #  初始位置应该在原点附近,或者根据你的模型进行调整.
    object_tube.set_pos((0, 0, 0.1))  # 稍微抬高一点,以便观察
    object_tube.set_rgba((0, 1, 0, 0.5))
    object_tube.attach_to(base)

    # 初始变换矩阵 (根据你的需求调整)
    T_initial = np.array([
        [1, 0, 0, 0.002693533897],
        [0, 1, 0, -0.008149208501],
        [0, 0, 1, 0.6390047073],
        [0, 0, 0, 1]
    ])

    # 获取 object_tube 的初始位姿(齐次变换矩阵)
    T_initial_object = object_tube.get_homomat()
    # 或者,如果你想基于 T_initial 设置 object_tube 的初始位置,可以使用: 
    # object_tube.set_homomat(T_initial)
    # T_initial_object = T_initial


    # 目标变换矩阵 (根据你的需求调整)
    T_transform = np.array([
        [0.3547237813, -0.9132992625,  0.2001387477,  0.02930317074],
        [-0.09771322459, 0.1766738147,  0.9794072509, -0.02930317074],
        [-0.9298511147, -0.3669753075, -0.02657106891,  0.02930317074],
        [0, 0, 0, 1]
    ])


    T_transform_object = np.dot(T_transform, T_initial_object) # 先应用初始位姿,再应用变换



    pos_list = np.linspace(T_initial_object[:3, 3], T_transform_object[:3, 3], 300)


    rot_list = [np.eye(3) for _ in pos_list]

    # 将位置和旋转组合成齐次变换矩阵列表
    box_conf_list = [rm.homomat_from_posrot(pos, rot) for pos, rot in zip(pos_list, rot_list)]

    counter = [0]  # 用于追踪当前帧的计数器
    flag = [0]     # 用于控制动画方向的标志


    def update(box_conf_list, counter, flag, task):
        """
        更新函数,用于在每一帧更新 object_tube 的位置和姿态.
        """
        object_tube.set_homomat(box_conf_list[counter[0]])
        #  不需要每次都 attach_to,因为 object_tube 已经在场景中了
        # object_tube.attach_to(base)

        if flag[0] == 0:
            counter[0] += 1
            if counter[0] >= len(box_conf_list) - 1:
                flag[0] = 1  # 改变方向
        else:
            counter[0] -= 1
            if counter[0] <= 0:
                flag[0] = 0  # 改变方向

        return task.again  # 继续执行下一帧


    # 使用 taskMgr 安排更新任务
    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[box_conf_list, counter, flag],
                          appendTask=True)

    base.run()
