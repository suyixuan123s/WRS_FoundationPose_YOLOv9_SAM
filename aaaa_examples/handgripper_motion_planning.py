import math
import numpy as np
import robot_sim.end_effectors.handgripper.dexgripper.dexgripper as cbt
import visualization.panda.world as wd
import modeling.geometric_model as gm

if __name__ == '__main__':

    base = wd.World(cam_pos=[1, 1, 0.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    robot_s = cbt.Handgripper()
    # robot_s.gen_meshmodel().attach_to(base)
    start_conf = robot_s.get_jnt_values(component_name='rgtfinger')
    tgt_pos = robot_s.get_gl_tcp("rgtfinger")[0]
    tgt_rotmat = robot_s.get_gl_tcp("rgtfinger")[1]
    jnt_values_list = []


    def get_path(start_joint, end_joint, moveinterval):
        jnt_values = start_conf + start_joint
        jnt_values_list.append(jnt_values)
        robot_s.fk(component_name="rgtfinger", jnt_values=jnt_values)
        pos_start, rot_start = robot_s.get_gl_tcp(manipulator_name="rgtfinger")
        robot_s.gen_meshmodel(rgba_lftfinger=(1, 0, 0, 1)).attach_to(base)
        print(pos_start)
        print('//')
        print(rot_start)
        print('//')
        jnt_values = start_conf + end_joint
        robot_s.fk(component_name="rgtfinger", jnt_values=jnt_values)
        pos_end, rot_end = robot_s.get_gl_tcp(manipulator_name="rgtfinger")
        robot_s.gen_meshmodel(rgba_lftfinger=(0, 0, 1, 1)).attach_to(base)
        print(pos_end)
        print('//')
        print(rot_end)
        print('//')
        stepwisepath = np.linspace(pos_start, pos_end, moveinterval, endpoint=False)
        color_p = 1
        goal_jnt_values = start_conf + np.array([math.pi * 1 / 4, math.pi * 1 / 4, -math.pi * 0 / 2])
        for path in stepwisepath:
            goal_jnt_values = robot_s.ik(tgt_pos=path, tgt_rotmat=rot_end, seed_jnt_values=goal_jnt_values,
                                         component_name="rgtfinger")
            jnt_values_list.append(goal_jnt_values)
            transparency = color_p / (moveinterval)
            print(transparency)
            color_p += 1
            robot_s.fk("rgtfinger", jnt_values=goal_jnt_values)
            robot_s.gen_meshmodel(rgba_rgtfinger=(0, 1, 0, transparency)).attach_to(base)
        jnt_values_list.append(end_joint)
        return jnt_values_list


    path = get_path(start_joint=np.array([math.pi * 1 / 4, math.pi * 1 / 4, -math.pi * 0 / 2]),
                    end_joint=np.array([0, 0, math.pi * 1 / 2]), moveinterval=10)
    print(path)
    base.run()
