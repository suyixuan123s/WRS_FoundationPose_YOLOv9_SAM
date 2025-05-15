import math
import numpy as np
import robot_sim.end_effectors.handgripper.dexgripper.dexgripper_smallitems as cbt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm

if __name__ == '__main__':

    base = wd.World(cam_pos=[1, 1, 0.5], lookat_pos=[0, 0, .2])
    gm.gen_frame(length=0.2).attach_to(base)
    gripper_s = cbt.Handgripper()
    gripper_s.preparation_grabbing()
    # gripper_s.gen_meshmodel().attach_to(base)
    f_pos, f_rot = gripper_s.get_gl_tcp("rgtfinger")
    # gm.gen_frame(f_pos, f_rot).attach_to(base)
    T_g_f = np.linalg.inv(rm.homomat_from_posrot(f_pos, f_rot))

    gm.gen_box([0.1, 0.1, 0.6], rgba=[0.2, 0.2, 0.2, 0.2]).attach_to(base)
    gm.gen_box([1, 1, 0.06], rm.homomat_from_posrot([0, 0, 0.3]), rgba=[0.2, 0.2, 0.2, 0.5]).attach_to(base)
    gm.gen_frame(pos=[0, 0, 0.33]).attach_to(base)
    target_pos = [0.1, 0.1, 0.33]
    rot_list = np.linspace(0, 360, 1, endpoint=False)
    for i in rot_list:
        transition_rot = rm.rotmat_from_axangle([0, 0, 1], np.deg2rad(i))
        transpose_y = np.array([[-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])
        rel_rot = np.dot(transition_rot, transpose_y)
        T_goal = rm.homomat_from_posrot(target_pos, rel_rot)
        gm.gen_sphere(pos=target_pos, radius=0.01).attach_to(base)
        # gm.gen_frame(pos = target_pos, rotmat=rel_rot).attach_to(base)
        print(np.dot(T_goal, T_g_f)[:3, 3], np.dot(T_goal, T_g_f)[:3, :3])
        gripper_s.move_to(np.dot(T_goal, T_g_f)[:3, 3], np.dot(T_goal, T_g_f)[:3, :3])
        gripper_s.gen_meshmodel(rgba_lftfinger=[0, 0.3, 0, 0.3], rgba_rgtfinger=[0, 0.3, 0, 0.3],
                                rgba_base=[0, 0.3, 0, 0.3]).attach_to(base)
    base.run()
