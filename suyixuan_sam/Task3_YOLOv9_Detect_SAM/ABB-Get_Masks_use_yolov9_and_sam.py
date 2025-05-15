# -*- coding: utf-8 -*-

"""

Author: Yixuan Su
Date: 2024/11/21 12:08
File: YOLOv9_detect_SAM_single_complete_image_saved2_mask_yolodetect_API.py
Description: 使用 YOLOv9 API 进行目标检测，并结合 SAM 提取掩码信息

"""
from suyixuan_sam.Task1_YOLOv9_Detect.YOLOv9_Detect_API import DetectAPI
import shutil
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# SAM 模型加载路径
sam_checkpoint = r"/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM
/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# 结果保存路径
save_folder = r'/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM
/suyixuan_sam/Task3_YOLOv9_Detect_SAM/results_yolodetect_API'
mask_output_folder = os.path.join(save_folder, 'masks')  # 用于保存掩码图像的文件夹
yolo_detection_image_path = os.path.join(save_folder, 'yolo_detections.jpg')  # 保存 YOLO 检测结果的图像路径


def clear_folder(folder_path):
    """清空指定文件夹下的所有文件和子文件夹."""
    try:
        if os.path.exists(folder_path):  # 检查文件夹是否存在
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            print(f"成功清空文件夹: {folder_path}")
        else:
            print(f"文件夹不存在，跳过清空: {folder_path}")
    except Exception as e:
        print(f"清空文件夹 {folder_path} 失败: {e}")


# ----------------- SAM 目标分割 -----------------
def sam_segment(img0, bbox):
    # 初始化 SAM 模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # 根据模型类型加载 SAM 模型
    predictor = SamPredictor(sam)  # 创建 SAM 预测器

    # 转换图像颜色空间
    image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB
    predictor.set_image(image)  # 设置 SAM 预测器的图像

    # 将 YOLO 边界框传递给 SAM 进行分割
    masks, _, _ = predictor.predict(
        point_coords=None,  # 没有点提示
        point_labels=None,  # 没有点标签
        box=np.array(bbox),  # 使用边界框作为提示
        multimask_output=False  # 只输出一个掩码
    )

    return masks[0]  # 返回分割掩码


# ----------------- 保存目标掩码图像（白色物体，黑色背景）-----------------
def save_object_mask_on_black_background(img, masks, bboxes, labels, save_folder):
    """
    保存每个检测到的物体的掩码图像，物体显示为白色，背景为黑色，并保持原始图像大小。

    Args:
        img (numpy.ndarray): 原始图像。
        masks (list): SAM 生成的掩码列表。
        bboxes (list): YOLOv9 检测到的边界框列表。
        labels (list): 物体标签列表。
        save_folder (str): 保存掩码图像的文件夹路径。
    """
    os.makedirs(save_folder, exist_ok=True)

    for i, (mask, bbox, label) in enumerate(zip(masks, bboxes, labels)):
        # 创建一个与原始图像大小相同的黑色图像
        mask_img = np.zeros_like(img, dtype=np.uint8)

        # 将掩码区域设置为白色
        mask_area = mask > 0
        mask_img[mask_area] = [255, 255, 255]  # 白色

        # 保存掩码图像
        save_path = os.path.join(save_folder, f"mask_{i}_{label.split()[0]}.png")  # 使用物体名称作为文件名
        cv2.imwrite(save_path, mask_img)
        print(f"物体 {label} 的掩码图像已保存: {save_path}")


# ----------------- 保存带有 YOLO 检测结果的图像 -----------------
def save_yolo_detection_image(img, detections, save_path, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制 YOLO 检测到的边界框和标签，并保存图像。

    Args:
        img (numpy.ndarray): 原始图像。
        detections (list): YOLOv9 检测结果列表 (xyxy 格式)。
        save_path (str): 保存图像的路径。
        color (tuple): 边界框的颜色 (BGR 格式)。
        thickness (int): 边界框的粗细。
    """
    img_copy = img.copy()
    for *xyxy, conf, cls in detections:  # 解包检测结果
        x0, y0, x1, y1 = map(int, xyxy)  # 转换为整数
        class_name = detect_api.names[int(cls)]  # 获取类别名称
        confidence = float(conf)

        cv2.rectangle(img_copy, (x0, y0), (x1, y1), color, thickness)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img_copy, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    cv2.imwrite(save_path, img_copy)
    print(f"带有 YOLO 检测结果的图像已保存: {save_path}")


# ----------------- 主函数 -----------------

if __name__ == "__main__":
    # 初始化 YOLOv9 API
    weights = '/home/suyixuan/AI/Pose_Estimation/WRS_FoundationPose_YOLOv9_SAM
/weights/best.pt'  # YOLOv9 权重文件路径
    csv_path = 'detection_results.csv'  # CSV 文件保存路径
    detect_api = DetectAPI(weights=weights, csv_path=csv_path)

    # 清空文件夹
    clear_folder(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # 读取图像
    source = r'00000.jpg'  # 图像路径
    img = cv2.imread(source)
    img_list = [img]  # YOLOv9 API 接受图像列表

    # YOLOv9 检测
    img_detected, pred = detect_api.detect(img_list)  # 使用 YOLOv9 API 进行检测
    # print("pred", pred)

    # 保存 img_detected
    img_detected_path = os.path.join(save_folder, 'img_detected.jpg')  # 保存路径
    cv2.imwrite(img_detected_path, img_detected)
    print(f"img_detected 已保存: {img_detected_path}")

    # 提取检测结果
    detections = []
    for *xyxy, conf, cls in pred[0]:  # pred 是一个列表，其中包含一个张量
        detections.append((*xyxy, conf, cls))  # xyxy, confidence, class

    # 遍历每个目标,使用 SAM 分割并保存结果
    masks = []
    bboxes = []
    labels = []

    for idx, detection in enumerate(detections):
        try:
            x0, y0, x1, y1, conf, cls = detection  # 解包检测结果
        except ValueError as e:
            print(f"解包错误: {e}, detection: {detection}")
            continue  # 跳过当前检测结果

        bbox = [int(x0.cpu().numpy()), int(y0.cpu().numpy()), int(x1.cpu().numpy()), int(y1.cpu().numpy())]  # 创建边界框列表
        class_name = detect_api.names[int(cls)]  # 获取类别名称
        confidence = float(conf)

        print(f"检测到目标: {class_name}, 置信度: {confidence:.2f}, 边界框: {bbox}")

        # SAM 分割
        mask = sam_segment(img, bbox)

        # 保存信息用于完整图像保存
        masks.append(mask)
        bboxes.append(bbox)
        labels.append(f"{class_name} {confidence:.2f}")

    # 保存目标掩码图像
    save_object_mask_on_black_background(img, masks, bboxes, labels, mask_output_folder)

    # 保存带有 YOLO 检测结果的图像
    save_yolo_detection_image(img, detections, yolo_detection_image_path)

    print("处理完成！")
