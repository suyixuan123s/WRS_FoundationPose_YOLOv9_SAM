if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.gofa5.gofa51 as cbt
    import robot_con.cobotta.cobotta_x as cbtx
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import time

    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    robot_s = cbt.GOFA5()
    start_conf = robot_s.get_jnt_values(component_name='arm')
    jnt_values = start_conf + np.array([0, 0.3, 0, 0.1, 0.5, 0.5])
    robot_s.fk(component_name="arm", jnt_values=jnt_values)
    robot_s.gen_meshmodel(rgba=(0, 1, 0, 1)).attach_to(base)
    tgt_pos = robot_s.get_gl_tcp("arm")[0]
    tgt_rotmat = robot_s.get_gl_tcp("arm")[1]

    goal_pos = tgt_pos + np.array([0, 0, -0.3])
    goal_rotmat = tgt_rotmat
    goal_jnt_values = robot_s.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    robot_s.fk(component_name="arm", jnt_values=goal_jnt_values)
    robot_s.gen_meshmodel(rgba=(0, 0, 1, 1)).attach_to(base)
    # base.run()
    print(jnt_values)
    print(goal_jnt_values)
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name="arm",
                             start_conf=jnt_values,
                             goal_conf=goal_jnt_values,
                             ext_dist=0.2,
                             max_time=300)
    print(path)
    for pose in path[1:-1]:
        robot_s.fk("arm", pose)
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
    base.run()
