import numpy as np
from numpy.linalg import inv
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
from panda3d.core import NodePath

if __name__ == '__main__':
    # 1. 创建 Panda3D 世界
    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0])

    table = cm.gen_box([2, 2, 0.001])
    table.set_rgba((0.51, 0.51, 0.51, 1))
    table.set_pos((0, 0, 0.095))
    table.attach_to(base)

    # 2. 定义物体相对于相机的转换矩阵 (假设已知)
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
    camera_model = gm.gen_box([0.05, 0.05, 0.05])
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

    # 11. 显示坐标轴
    gm.gen_frame().attach_to(base)

    # 12. 运行 Panda3D 世界
    base.run()
