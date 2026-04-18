"""
utils.py

功能: 测试辅助函数
时间: 2026/03/20
版本: 1.0
"""

import os
import copy
from typing import Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from PIL import Image


# def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
#     """计算单个样本的二分类 Dice 系数和 IoU"""
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     sum_pred = pred.sum()
#     sum_gt = gt.sum()
#     if sum_pred == 0 and sum_gt == 0:
#         return 1, 1
#     if sum_pred > 0 and sum_gt > 0:
#         intersection = np.sum(pred * gt)
#         union = sum_pred + sum_gt - intersection
#         dice = metric.binary.dc(pred, gt)
#         iou = intersection / union
#         return dice, iou
#     return 0, 0


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float, float]:
    """计算单个样本的指标"""
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    # dice 和 iou
    if TP + FP + FN == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2 * TP) / (2 * TP + FP + FN)
        iou = TP / (TP + FP + FN)

    # acc
    acc = (TP + TN) / (TP + TN + FP + FN)

    # recall
    if TP + FN == 0:
        rec = 1.0
    else:
        rec = TP / (TP + FN)

    # precision
    if TP + FP == 0:
        pre = 1.0
    else:
        pre = TP / (TP + FP)

    return dice, iou, acc, rec, pre


def test_single_volume(
    image: torch.Tensor,
    label: torch.Tensor,
    net: torch.nn.Module,
    split: str,
    classes: int,
    patch_size: List[int],
    test_save_path: Optional[str] = None,
    case: Optional[str] = None,
    z_spacing: int = 1
) -> List[Tuple[float, float, float, float, float]]:
    """对单个三维体数据进行推理并计算每个类别的指标"""
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    _, x, y = image.shape

    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()

    with torch.no_grad():
        out = torch.sigmoid(net(input)).squeeze(0).cpu().numpy()
        prediction = (out > 0.5).astype(np.uint8)   # 将 out 二分类
        if prediction.ndim == 3:
            prediction = prediction[0]

        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    label = (label > 0.5).astype(np.uint8)  # 将 label 二分类
    if label.ndim == 3:
        label = label[0]

    metric_list = [calculate_metric_percase(prediction, label)]   # 计算二分类的指标结果
    return metric_list

    # if test_save_path is not None:
    #     original_image = image.transpose(1, 2, 0)
    #     if original_image.shape[2] == 1:
    #         original_image = original_image.squeeze(2)
    #     original_image_normalized = (original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255
    #     original_image_normalized = original_image_normalized.astype(np.uint8)
    #
    #     # 保存原始图像
    #     original_image_path = os.path.join(test_save_path, f"{case}_original.png")
    #     cv2.imwrite(original_image_path, original_image_normalized)
    #
    #     # 创建一个新的图像用于显示标签叠加在原图像上
    #     label_on_image = original_image_normalized.copy()
    #
    #     # 将标签部分设置为浅红色
    #     label_on_image[label == 1] = [255, 200, 200]  # 浅红色
    #     label_on_image[label == 2] = [255, 200, 200]  # 浅红色
    #     label_on_image[label == 3] = [255, 200, 200]  # 浅红色
    #     label_on_image[label == 4] = [255, 200, 200]  # 浅红色
    #
    #     # 保存叠加后的图像
    #     overlay_image = Image.fromarray(label_on_image)
    #     overlay_image.save(os.path.join(test_save_path, f"{case}_overlay.png"))
    #
    #     # 保存标签图像
    #     a1_label = copy.deepcopy(label)
    #     a2_label = copy.deepcopy(label)
    #     a3_label = copy.deepcopy(label)
    #
    #     a1_label[a1_label == 1] = 255
    #     a1_label[a1_label == 2] = 0
    #     a1_label[a1_label == 3] = 255
    #     a1_label[a1_label == 4] = 20
    #
    #     a2_label[a2_label == 1] = 255
    #     a2_label[a2_label == 2] = 255
    #     a2_label[a2_label == 3] = 0
    #     a2_label[a2_label == 4] = 10
    #
    #     a3_label[a3_label == 1] = 255
    #     a3_label[a3_label == 2] = 77
    #     a3_label[a3_label == 3] = 0
    #     a3_label[a3_label == 4] = 120
    #
    #     a1_label = Image.fromarray(np.uint8(a1_label)).convert("L")
    #     a2_label = Image.fromarray(np.uint8(a2_label)).convert("L")
    #     a3_label = Image.fromarray(np.uint8(a3_label)).convert("L")
    #     label_rgb = Image.merge("RGB", [a1_label, a2_label, a3_label])
    #
    #     label_rgb.save(os.path.join(test_save_path, f"{case}_label.png"))
    #
    #     # 保存预测结果
    #     a1 = copy.deepcopy(prediction)
    #     a2 = copy.deepcopy(prediction)
    #     a3 = copy.deepcopy(prediction)
    #
    #     a1[a1 == 1] = 255
    #     a1[a1 == 2] = 0
    #     a1[a1 == 3] = 255
    #     a1[a1 == 4] = 20
    #
    #     a2[a2 == 1] = 255
    #     a2[a2 == 2] = 255
    #     a2[a2 == 3] = 0
    #     a2[a2 == 4] = 10
    #
    #     a3[a3 == 1] = 255
    #     a3[a3 == 2] = 77
    #     a3[a3 == 3] = 0
    #     a3[a3 == 4] = 120
    #
    #     a1 = Image.fromarray(np.uint8(a1)).convert("L")
    #     a2 = Image.fromarray(np.uint8(a2)).convert("L")
    #     a3 = Image.fromarray(np.uint8(a3)).convert("L")
    #     prediction_image = Image.merge("RGB", [a1, a2, a3])
    #     prediction_image.save(os.path.join(test_save_path, f"{case}_prediction.png"))
    #
    # return metric_list


def plot_img_and_mask(img: np.ndarray, mask: np.ndarray) -> None:
    """可视化输入图像及其对应的分割掩码"""
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f"Mask (class {i + 1})")
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
