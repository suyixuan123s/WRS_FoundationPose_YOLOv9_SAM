# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from estimater import *
from datareader import *
import argparse
from suyixuan_sam.Task3_YOLOv9_Detect_SAM.ABB_Get_Masks_use_yolov9_and_sam_hjy import main as abb_main
from camera_capture import capture_images


def estimate_pose(mesh_file, test_scene_dir, est_refine_iter, track_refine_iter, debug_level, debug_dir):
    """
    姿态估计函数
    :param rgb_image_path: RGB 图像路径
    :param depth_image_path: 深度图像路径
    :param mesh_file: 网格文件路径
    :param test_scene_dir: 测试场景目录
    :param est_refine_iter: 姿态估计细化迭代次数
    :param track_refine_iter: 姿态跟踪细化迭代次数
    :param debug_level: 调试级别
    :param debug_dir: 调试目录
    :return: 姿态估计结果
    """
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_file)

    debug = debug_level
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                         refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                                is_input_rgb=True)
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    return pose


if __name__ == '__main__':
    save_directory_rgb = r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/rgb'
    save_directory_depth = r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/depth'
    capture_images(save_directory_rgb, save_directory_depth, wait_time=5)
    rgb_image_path = os.path.join(save_directory_rgb, '000001.png')
    depth_image_path = os.path.join(save_directory_depth, '000001.png')
    while not (os.path.exists(rgb_image_path) and os.path.exists(depth_image_path)):
        print("rgb或depth没有保存")
        capture_images(save_directory_rgb, save_directory_depth, wait_time=5)
        time.sleep(1)
    print("rgb和depth都成功保存了!")
    abb_main(source='/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/rgb/000001.png',
             weights='/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/weights/osgj.pt',
             target_labels="j")

    # 参数解析
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/514/mesh/textured_mesh.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/514')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=3)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_data/514')
    args = parser.parse_args()

    # 调用姿态估计函数
    pose = estimate_pose(args.mesh_file, args.test_scene_dir, args.est_refine_iter,
                         args.track_refine_iter, args.debug, args.debug_dir)

    print("姿态估计结果:\n", pose)
