import copy
import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh
import robot_sim.robots.gofa5.gofa51 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)

    rbt_s = gf5.GOFA5()
    rbt_s.hnd.jaw_to(.060)

    # rbt_s.hnd.open()
    rbt_s.gen_meshmodel().attach_to(base)

    manipulator_name = "arm"
    start_conf = rbt_s.get_jnt_values(manipulator_name)
    # start_conf = np.array([0.0439823 , -0.53023103  ,1.05243354 , 0.0143117  , 1.55351757 , 1.57079633])
    print(start_conf)
    hand_name = "hnd"

    # object
    rack_name = "rack_10ml_new"
    rack_10ml = cm.CollisionModel(f"objects/{rack_name}.stl")
    rack_10ml.set_rgba([1, 1, 1, 1])
    rack_10ml.set_pos(np.array([.6, -.5, -0.015]))
    rack_10ml.set_rotmat()
    rack_10ml.attach_to(base)

    tube_name = "blood_tube10"
    tube1 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube1.set_rgba([.0, 1, .0, 1])
    tube1.set_pos(np.array([.555, -.5, 0.098]))
    tube1.set_rotmat()
    tube1.attach_to(base)

    tube2 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube2.set_rgba([.0, 1, .0, 1])
    tube2.set_pos(np.array([.585, -.5, 0.098]))
    tube2.set_rotmat()
    tube2.attach_to(base)

    tube3 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube3.set_rgba([1, 0, 0, 1])
    tube3.set_pos(np.array([.615, -.5, 0.098]))
    tube3.set_rotmat()
    tube3.attach_to(base)

    tube4 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube4.set_rgba([.0, 1, .0, 1])
    tube4.set_pos(np.array([.645, -.5, 0.098]))
    tube4.set_rotmat()
    tube4.attach_to(base)

    tube5 = cm.CollisionModel(f"objects/{tube_name}.stl")
    tube5.set_rgba([.0, 1, .0, 1])
    tube5.set_pos(np.array([.645, -.475, 0.098]))
    tube5.set_rotmat()
    tube5.attach_to(base)

    base.run()
