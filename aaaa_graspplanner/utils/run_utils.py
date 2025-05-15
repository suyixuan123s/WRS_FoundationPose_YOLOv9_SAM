import copy
import os
import pickle

import cv2
import numpy as np

import config
import modeling.collision_model as cm
import localenv.item as item
import motionplanner.motion_planner as m_planner
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.prj_utils as pu
import utils.vision_utils as vu
from localenv import envloader as el


# import db_service.db_service as dbs
# import motionplanner.rbtx_motion_planner as m_plannerx
# from nailseg import nailseg as ns

def load_phxiinfo(phoxi_f_name, load=True, toggledebug=False):
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    f_path = os.path.join("img", phoxi_f_name)
    if load:
        grayimg, depthnparray_float32, pcd = phxi.loadalldata(f_path)
    else:
        grayimg, depthnparray_float32, pcd = phxi.dumpalldata(f_path)
        with open(os.path.join(config.ROOT, f_path), 'wb') as f:
            pickle.dump([grayimg, depthnparray_float32, pcd], f)
    if toggledebug:
        print("img size:", grayimg.shape)
        print("pcd size:", pcd.shape)
        cv2.imshow("load_phxiinfo", grayimg)
        cv2.waitKey(0)
    return grayimg, depthnparray_float32, np.array(pcd)


def get_obj_by_range(phxilocator, load=True, phoxi_f_name=None, reconstruct_surface=False, resolution=1,
                     x_range=(400, 1080), y_range=(-100, 300), z_range=(790, 1000), sample_num=None):
    grayimg, depthnparray_float32, pcd = load_phxiinfo(phoxi_f_name, load=load)
    objpcd_list = phxilocator.find_objpcd_list_by_pos(pcd, scan_num=resolution, toggledebug=False,
                                                      x_range=x_range, y_range=y_range, z_range=z_range)
    if len(objpcd_list) == 0:
        print("No obj is detected!")
        return None
    else:
        objpcd = phxilocator.find_largest_objpcd(objpcd_list)
        if sample_num is None:
            return item.Item(pcd=objpcd, reconstruct=reconstruct_surface)
        else:
            objcm = pcdu.reconstruct_surface(objpcd, radii=[5])
            return item.Item(reconstruct=reconstruct_surface, sample_num=sample_num, objcm=objcm)


def get_obj_from_phoxiinfo_withmodel_rmbg(phxilocator, stl_f_name, load_f_name=None, match_filp=False,
                                          bg_f_name="bg_0217.pkl"):
    objcm = cm.CollisionModel(objinit=os.path.join(config.ROOT + '/obstacles/' + stl_f_name))
    grayimg, depthnparray_float32, pcd = load_phxiinfo(phoxi_f_name=load_f_name)

    workingarea_uint8 = phxilocator.remove_depth_bg(depthnparray_float32, bg_f_name=bg_f_name, toggledebug=False)
    obj_depth = phxilocator.find_paintingobj(workingarea_uint8)

    if obj_depth is None:
        print("painting object not detected!")

    objpcd = pcdu.trans_pcd(pcdu.remove_pcd_zeros(vu.map_depth2pcd(obj_depth, pcd)), phxilocator.amat)
    objmat4 = phxilocator.match_pcdncm(objpcd, objcm, match_rotz=match_filp)
    objcm.set_homomat(objmat4)

    return item.Item(objcm=objcm, pcd=objpcd, objmat4=objmat4)


def get_obj_from_phoxiinfo_withmodel(phxilocator, stl_f_name, objpcd_list=None, phoxi_f_name=None, load=True,
                                     x_range=(200, 800), y_range=(-100, 300), z_range=(790, 1000),
                                     match_rotz=False, resolution=1, eps=5, use_rmse=True):
    if stl_f_name[-3:] != 'stl':
        stl_f_name += '.stl'
    objcm = cm.CollisionModel(initor=os.path.join(config.ROOT + '/obstacles/' + stl_f_name))
    if objpcd_list is None:
        grayimg, depthnparray_float32, pcd = load_phxiinfo(phoxi_f_name=phoxi_f_name, load=load)
        objpcd_list = phxilocator.find_objpcd_list_by_pos(pcd, x_range=x_range, y_range=y_range, z_range=z_range,
                                                          toggledebug=False, scan_num=resolution, eps=eps)

    if len(objpcd_list) == 0:
        print("No obj is detected!")

    objpcd = phxilocator.find_closest_objpcd_by_stl(stl_f_name, objpcd_list, use_rmse=use_rmse)
    objmat4 = phxilocator.match_pcdncm(objpcd, objcm, toggledebug=False, match_rotz=match_rotz)
    # objmat4[:3, :3] = np.eye(3)
    # objmat4[:3, 3] = objmat4[:3, 3] + np.asarray([-20, 0, 0])
    objcm.set_homomat(objmat4)

    return item.Item(objcm=objcm, pcd=objpcd, objmat4=objmat4)


def get_obj_inhand_from_phoxiinfo_withmodel(phxilocator, stl_f_name, tcp_pos, inithomomat=None, phoxi_f_name=None,
                                            load=True, showicp=False, showcluster=False):
    """
    find obj pcd directly by clustering

    :param phxilocator:
    :param stl_f_name:
    :param tcp_pos:
    :param inithomomat:
    :param phoxi_f_name:
    :param temp_f_name:
    :param w:
    :param h:
    :return:
    """
    objcm = cm.CollisionModel(initor=os.path.join(config.ROOT + '/obstacles/' + stl_f_name))
    grayimg, depthnparray_float32, pcd = load_phxiinfo(phoxi_f_name=phoxi_f_name, load=load)

    objpcd = phxilocator.find_objinhand_pcd(tcp_pos, pcd, stl_f_name, toggledebug=showcluster)
    objmat4 = phxilocator.match_pcdncm(objpcd, objcm, inithomomat, toggledebug=showicp)
    objcm.set_homomat(objmat4)

    return item.Item(objcm=objcm, pcd=objpcd, objmat4=objmat4, draw_center=tcp_pos)


def get_pen_objmat4_list_by_drawpath(drawpath, paintingobj_item, drawrec_size=(.015, .015), space=0,
                                     color=(1, 0, 0), mode="DI", direction=np.asarray((0, 0, 1)), show=True):
    drawpath = pu.resize_drawpath(pu.remove_list_dup(drawpath), drawrec_size[0], drawrec_size[1], space)
    paintingobj_objmat4 = paintingobj_item.objmat4

    if mode == "rh":
        # rayhit
        paintingobj_cm = copy.deepcopy(paintingobj_item.objcm)
        paintingobj_cm.sethomomat(paintingobj_objmat4)
        pos_nrml_list, error_list = pu.rayhitmesh_drawpath(paintingobj_item, drawpath)

    elif mode in ['DI', 'EI', 'QI', 'rbf', 'rbf_g', 'gaussian', 'quad', 'bs', 'bp']:
        # metrology
        pos_nrml_list, error_list, time_cost = \
            pu.prj_drawpath(paintingobj_item, drawpath, mode=mode, direction=direction)
    else:
        # conformal mapping
        pos_nrml_list, error_list = pu.prj_drawpath_II(paintingobj_item, drawpath)
    objmat4_draw_list_ms = []
    for sub_list in pos_nrml_list:
        if show:
            pu.show_drawpath(sub_list, color=color)
        objmat4_draw_list_ms.append(pu.get_penmat4(sub_list))

    return objmat4_draw_list_ms


def get_objmat4_posdiff(objmat4, objmat4_real):
    return objmat4_real[:3, 3] - objmat4[:3, 3]


def get_objmat4_list_by_posdiff(objmat4_list, posdiff):
    objmat4_real_list = []
    for objmat4 in objmat4_list:
        objmat4_nxt = copy.deepcopy(objmat4)
        objmat4_nxt[:3, 3] = objmat4_nxt[:3, 3] + posdiff
        objmat4_real_list.append(objmat4_nxt)
    return objmat4_real_list


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt= el.loadUr3e()
    rbtx = el.loadUr3ex

    pen_stl_f_name = "pentip"

    '''
    init planner
    '''
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    obj = cm.CollisionModel(initor=os.path.join(config.ROOT + '/obstacles/pentip.stl'))

    objmat4_list = pickle.load(
        open(config.GRASPMAP_REL_PATH + pen_stl_f_name + "_objmat4_list.pkl", "rb"))
    grasp_map = pickle.load(open(config.GRASPMAP_REL_PATH + pen_stl_f_name + "_graspmap.pkl", "rb"))
    print(len(objmat4_list))
    objmat4_id_list = []
    for grasp_id, v in grasp_map.items():
        for objmat4_id, bool in v.items():
            if bool:
                objmat4_id_list.append(objmat4_id)
    objmat4_id_list = list(set(objmat4_id_list))
    print(len(objmat4_id_list), objmat4_id_list)
    for id in range(len(objmat4_list)):
        if id in objmat4_id_list:
            motion_planner_lft.ah.show_objmat4(obj, objmat4_list[id], rgba=(0, 1, 0, .5))
        else:
            motion_planner_lft.ah.show_objmat4(obj, objmat4_list[id], rgba=(1, 0, 0, .5))
    base.run()
