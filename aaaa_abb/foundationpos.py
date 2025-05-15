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
    base.run()

    # pos_list = np.linspace([0, 0, 0.01], [1, 0, 0.01], 300)
    pos_list = np.linspace([0.003, -0.008, 0.639],
                           [0.002, -0.008, 0.641], 300)

    rot_list = [np.eye(3) for item in pos_list]
    box_conf_list = [rm.homomat_from_posrot(pos_list[i], rot_list[i]) for i in range(len(pos_list))]

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
