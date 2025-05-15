# 这段代码实现了一个简单的点云可视化程序,使用 RealSense D400 相机获取实时的点云数据,并在 Panda3D 环境中进行可视化.
# 通过不断更新点云数据,用户可以实时观察相机捕获的三维场景.整体结构清晰,适合用于机器人视觉和三维数据处理的应用.

import wrs.drivers.devices.realsense.realsense_d400s as rs
from wrs import wd, rm, mgm

base = wd.World(cam_pos=rm.vec(2, 1, 1), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)

d405 = rs.RealSenseD400()
onscreen = []


def update(d405, onscreen, task):
    if len(onscreen) > 0:
        for ele in onscreen:
            ele.detach()
    pcd, pcd_color = d405.get_pcd(return_color=True)
    onscreen.append(mgm.gen_pointcloud(pcd, pcd_color))
    onscreen[-1].attach_to(base)
    return task.cont


base.taskMgr.add(update, "update", extraArgs=[d405, onscreen], appendTask=True)
base.run()
