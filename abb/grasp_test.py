import time
import os
import pickle

import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.end_effectors.gripper.dh76.dh76 as dh76
import robot_sim.robots.gofa5.gofa5 as gf5
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import motion.optimization_based.incremental_nik as inik
import basis.robot_math as rm
import robot_con.gofa_con.gofa_con as gofa_con
import drivers.devices.dh.maingripper as dh_r
from direct.task.TaskManagerGlobal import taskMgr

def save_pickle_data(name, data):
    dir_name = os.path.dirname(name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(name, "wb") as file:
        pickle.dump(data, file)

def load_pickle_data(name):
    with open(name, "rb") as file:
        f = pickle.load(file)
    return f

def load_grasp_list(pair_type, debug_folder):
    grip_info_path = f"{debug_folder}/grip_info_dict.pickle"
    pair_stability_path = f"{debug_folder}/pair_stability.pickle"
    pair_type_path = f"{debug_folder}/pair_type.pickle"
    if os.path.exists(grip_info_path) and os.path.exists(pair_stability_path) and os.path.exists(
            pair_type_path):
        grip_info_dict = load_pickle_data(f'{debug_folder}/grip_info_dict.pickle')
        pair_stability = load_pickle_data(f'{debug_folder}/pair_stability.pickle')
        pair_types = load_pickle_data(f'{debug_folder}/pair_type.pickle')
    else:
        raise FileNotFoundError("Grasp info file not found!")
    grasp_list = []
    jaw_width_list = grip_info_dict['jaw_width']
    jaw_center_pos = grip_info_dict['jaw_center_pos']
    jaw_center_rotmat = grip_info_dict['jaw_center_rotmat']
    hand_pos = grip_info_dict['hand_pos']
    hand_rotmat = grip_info_dict['hand_rotmat']
    if pair_type == 'all':
        pair_grip_info = [np.array(jaw_width_list).reshape(-1, 1),
                          np.array(jaw_center_pos).reshape(-1, 3),
                          np.array(jaw_center_rotmat).reshape(-1, 3, 3),
                          np.array(hand_pos).reshape(-1, 3),
                          np.array(hand_rotmat).reshape(-1, 3, 3)]
    else:
        if pair_type not in ['cc', 'ce', 'cf', 'ee', 'ef', 'ff']:
            raise Exception("Invalid pair type!")
        else:
            pair_id = np.where((np.array(pair_types) == pair_type))[0]
            pair_grip_info = [np.array(jaw_width_list)[pair_id].reshape(-1, 1),
                              np.array(jaw_center_pos)[pair_id].reshape(-1, 3),
                              np.array(jaw_center_rotmat)[pair_id].reshape(-1, 3, 3),
                              np.array(hand_pos)[pair_id].reshape(-1, 3),
                              np.array(hand_rotmat)[pair_id].reshape(-1, 3, 3)]
    for i in range(pair_grip_info[0].shape[0]):
        row = [pair_grip_info[0][i], pair_grip_info[1][i], pair_grip_info[2][i],
               pair_grip_info[3][i], pair_grip_info[4][i]]
        grasp_list.append(row)

    return grasp_list

def load_single_grasp(pair_type, pair_id, rot_id, jaw_width_ratio):
    grip_info_path = f"{debug_folder}/grip_info_dict.pickle"
    pair_stability_path = f"{debug_folder}/pair_stability.pickle"
    pair_type_path = f"{debug_folder}/pair_type.pickle"
    if os.path.exists(grip_info_path) and os.path.exists(pair_stability_path) and os.path.exists(pair_type_path):
        grip_info_dict = load_pickle_data(f'{debug_folder}/grip_info_dict.pickle')
        pair_stability = load_pickle_data(f'{debug_folder}/pair_stability.pickle')
        pair_types = load_pickle_data(f'{debug_folder}/pair_type.pickle')
    else:
        raise FileNotFoundError("Grasp info file not found!")
    indices = np.where(np.array(pair_types) == pair_type)[0]
    i = indices[pair_id]
    single_grasp = [grip_info_dict['jaw_width'][i][rot_id] * jaw_width_ratio,
                    grip_info_dict['jaw_center_pos'][i][rot_id],
                    grip_info_dict['jaw_center_rotmat'][i][rot_id],
                    grip_info_dict['hand_pos'][i][rot_id],
                    grip_info_dict['hand_rotmat'][i][rot_id]]
    return single_grasp


if __name__ == '__main__':
    rbt_r = gofa_con.GoFaArmController(toggle_debug=False)
    dh76_con = dh_r.MainGripper(port="com4")
    dh76_con.init_gripper()
    base = wd.World(cam_pos=[4.16951, 1.8771, 1.70872], lookat_pos=[0, 0, 0.5])
    obj_name = 'lblock_c'
    debug_folder = f'./debug_data/{obj_name}'
    obj_path = f"./test_obj/{obj_name}.stl"
    obj = cm.CollisionModel(obj_path)
    obj.set_pos((0.6, 0, -0.014 + 0.038))
    obj.attach_to(base)
    rbt_s = gf5.GOFA5()
    gripper_s = dh76.Dh76(fingertip_type='r_76',
                          pos=rbt_s.arm.jnts[-1]['gl_posq'],
                          rotmat=rbt_s.arm.jnts[-1]['gl_rotmatq'],
                          name='hnd', enable_cc=False)
    rbt_s.jaw_to('hnd', 0.06)
    # rbt_s.hnd = gripper_s
    # rbt_s.hnd_dict['hnd'] = gripper_s
    # rbt_s.hnd_dict['arm'] = gripper_s

    # grasp_info = load_single_grasp('ff', 3, 6, 0.8)
    grasp_info_list = load_grasp_list('ff', debug_folder)
    rbt_s.gen_meshmodel().attach_to(base)

    # a = gripper_s.gen_meshmodel()
    a_homo = rm.homomat_from_posrot((0.6, 0, -0.014 + 0.038))
    target_conf_list = []
    approach_conf_list = []
    for grasp_info in grasp_info_list:
        grasp_homo = rm.homomat_from_posrot(grasp_info[1], grasp_info[2])
        b = a_homo.dot(grasp_homo)

        # wrist_homo = rm.homomat_from_posrot(grasp_info[3], grasp_info[4])
        # c = a_homo.dot(wrist_homo)
        gripper_s.grip_at_with_jcpose(b[:3, 3], b[:3, :3], grasp_info[0])
        # gripper_s.gen_meshmodel().attach_to(base)
        try:
            # gm.gen_sphere(c[:3,3]).attach_to(base)
            target_conf = rbt_s.ik(component_name="arm",
                                   tgt_pos=b[:3, 3],
                                   tgt_rotmat=b[:3, :3],
                                   seed_jnt_values=rbt_s.get_jnt_values('arm'),
                                   max_niter=200,
                                   tcp_jnt_id=None,
                                   tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   local_minima="end",
                                   toggle_debug=False)
            rbt_s.fk('arm', target_conf)
            if rbt_s.is_collided():
                pass
            else:
                target_conf_list.append(target_conf)
                # rbt_s.gen_meshmodel(rgba=[0,1,0,0.3]).attach_to(base)
                pos, rot = rbt_s.get_gl_tcp("arm")
                pos_a = pos + [0, 0, 0.2]
                try:
                    approach_conf = rbt_s.ik('arm', pos_a, rot, seed_jnt_values=target_conf)
                    approach_conf_list.append([approach_conf, target_conf])
                    rbt_s.fk('arm', approach_conf)
                    # rbt_s.gen_meshmodel(rgba=[0,0,1,1]).attach_to(base)
                except:
                    pass
        except:
            pass
    # self.rrtc_s = rrtc.RRTConnect(self.rbt_s)
    # self.inik_s = inik.IncrementalNIK(self.rbt_s)

    init_jnts = np.array([0.0, .0, 0.0, 0.0, 0.0, 0.0])
    rrtc_s = rrtc.RRTConnect(rbt_s)

    for approach_conf in approach_conf_list:
        path_app = rrtc_s.plan('arm',
                               init_jnts,
                               approach_conf[0],
                               obstacle_list=[obj],
                               otherrobot_list=[],
                               ext_dist=0.03,
                               max_iter=300,
                               max_time=15.0,
                               smoothing_iterations=50,
                               animation=False)
        if path_app is not None:
            path_gri = rrtc_s.plan('arm',
                                   approach_conf[0],
                                   approach_conf[1],
                                   obstacle_list=[obj],
                                   otherrobot_list=[],
                                   ext_dist=0.03,
                                   max_iter=300,
                                   max_time=15.0,
                                   smoothing_iterations=50,
                                   animation=False)
            if path_gri is not None:
                # rbt_s.fk('arm', approach_conf[1])
                # rbt_s.gen_meshmodel().attach_to(base)
                # rbt_s.fk('arm', approach_conf[0])
                # rbt_s.gen_meshmodel().attach_to(base)

                # for item in path_app[::-1]:
                #     rbt_s.fk('arm', item)
                #     rbt_s.gen_meshmodel(rgba=[0,0,1,0.5]).attach_to(base)
                #
                # for item in path_gri[::-1]:
                #     rbt_s.fk('arm', item)
                #     rbt_s.gen_meshmodel(rgba=[0,1,0,0.5]).attach_to(base)

                rbt_r.move_jntspace_path(path_app)
                rbt_r.move_jntspace_path(path_gri)
                dh76_con.jaw_to(0)
                time.sleep(2)
                rbt_r.move_jntspace_path(path_gri[::-1])
                rbt_r.move_jntspace_path(path_gri)
                dh76_con.jaw_to(0.06)
                rbt_r.move_jntspace_path(path_gri[::-1])
                rbt_r.move_jntspace_path(path_app[::-1])
                break
    # rbt_r = gofa_con.GoFaArm()

    base.run()
