import os
from direct.task.TaskManagerGlobal import taskMgr
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import numpy as np


if __name__ == '__main__':
    base = wd.World(cam_pos=[-0.00269353, 0.00814921, -0.63900471], lookat_pos=[0, 0, .9])
    gm.gen_frame().attach_to(base)
    table = cm.gen_box([2, 1, 0.001])
    table.set_rgba((0.118, 0.275, 0.510, 1))
    table.set_pos((0, 0, 0.0005))

    object_tube = cm.CollisionModel("textured_mesh.obj")
    object_tube.set_pos()
    object_tube.set_rgba((0, 1, 0, 0.5))

    table.attach_to(base)
    object_tube.attach_to(base)
    # base.run()

    # 文件夹路径
    folder_path = r"E:\ABB-Project\ABB_wrs\suyixuan\ABB\data\out_rack\ob_in_cam"

    # 获取所有 txt 文件,并按文件名排序
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    # 仅筛选文件名在 680 到 710 之间的
    selected_files = [f for f in file_list if 680 <= int(f.split(".")[0]) <= 710]

    # 读取变换矩阵
    box_conf_list = []
    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as f:
            lines = f.readlines()

        # 解析矩阵
        matrix = np.array([[float(num) for num in line.split()] for line in lines])

        # 确保是 4x4 变换矩阵
        if matrix.shape == (4, 4):
            box_conf_list.append(matrix)
        else:
            print(f"Warning: {file_name} has an invalid format!")

    print(f"Loaded {len(box_conf_list)} transformation matrices.")

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


    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[box_conf_list, counter, flag],
                          appendTask=True)

    base.run()
