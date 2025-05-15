# -*- coding: utf-8 -*-

"""
Author: Yixuan Su
Date: 2025/05/14 17:16
File: camera_capture.py
Description:

"""


import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from PIL import Image
import glob


def capture_images(save_directory_rgb, save_directory_depth,wait_time=5):
    """
    从 Intel Realsense D435 相机捕获图像和深度数据，并保存到指定目录。
    在拍摄前等待指定的时间。

    Args:
        save_directory (str): 保存图像的目录。
        wait_time (int): 拍摄前等待的时间，单位为秒。默认为 5 秒。
    """
    # 确保目录存在
    os.makedirs(save_directory_rgb, exist_ok=True)
    os.makedirs(save_directory_depth, exist_ok=True)


    # 初始化管道
    pipeline = rs.pipeline()

    config = rs.config()

    # 配置相机流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # 启动管道
    pipeline.start(config)

    # 设置对齐器,将深度图与彩色图像对齐
    align_to = rs.stream.color  # 对齐到颜色流
    align = rs.align(align_to)

    try:
        print(f"等待 {wait_time} 秒后拍摄...")
        time.sleep(wait_time)  # 等待指定的时间

        while True:
            # 获取一帧数据
            frames = pipeline.wait_for_frames()

            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)

            # 获取对齐后的彩色帧和深度帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将图像转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 显示彩色图像
            cv2.imshow('Color Image', color_image)

            # 按 Enter 键保存RGB图像和深度图像
            key = cv2.waitKey(1)
            if key == 13:  # Enter key
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                for file in glob.glob(os.path.join(save_directory_rgb, '*.png')):
                    try:
                        os.remove(file)
                        print(f'Removed {file}')
                    except Exception as e:
                        print(f'Error removing {file}: {e}')

                # 保存RGB图像
                color_image_path = os.path.join(save_directory_rgb, f'000001.png')
                cv2.imwrite(color_image_path, color_image)
                print(f'Saved color image as {color_image_path}')

                for file in glob.glob(os.path.join(save_directory_depth, '*.png')):
                    try:
                        os.remove(file)
                        print(f'Removed {file}')
                    except Exception as e:
                        print(f'Error removing {file}: {e}')

                # 保存深度图像 (使用 Pillow)
                depth_image_path = os.path.join(save_directory_depth, f'000001.png')
                depth_image = depth_image.astype(np.uint16)  # 确保深度图像是 uint16 类型
                img = Image.fromarray(depth_image)
                img.save(depth_image_path)
                print(f'Saved depth image as {depth_image_path}')
                return color_image_path, depth_image_path  # 返回图像路径

            # 按 'q' 键退出
            if key & 0xFF == ord('q'):
                break
    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
        return None, None


if __name__ == "__main__":
    save_directory_rgb= r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/rgb'
    save_directory_depth = r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM/demo_data/514/depth'
    capture_images(save_directory_rgb, save_directory_depth,wait_time=5)
