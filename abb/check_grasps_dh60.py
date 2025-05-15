import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.dh60.dh60 as dh60

if __name__ == '__main__':
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    gripper_s = dh60.Dh60()
    objcm_name = "rack_10ml_new"
    obj = cm.CollisionModel(f"objects/{objcm_name}.stl")
    obj.set_rgba([.9, .75, .35, 1])
    obj.attach_to(base)
    obj.show_localframe()
    # base.run()

    # 从pickle文件中加载与物体相关的抓取信息
    grasp_info_list = gpa.load_pickle_file(objcm_name, root=None, file_name='dh60_grasps.pickle')
    for grasp_info in grasp_info_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        # 使用夹爪模型执行抓取,设置夹爪的位置、旋转矩阵和抓取宽度
        gripper_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
        gripper_s.gen_meshmodel().attach_to(base)
        print(jaw_width)
    base.run()
