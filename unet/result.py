import os
import numpy as np
from PIL import Image
import cv2  # 新增OpenCV预处理


def calculate_iou_dice(mask_gt, mask_pred, epsilon=1e-6):
    """优化计算逻辑，增加数值稳定性"""
    intersection = np.logical_and(mask_gt, mask_pred).sum().astype(float)
    union = np.logical_or(mask_gt, mask_pred).sum().astype(float)

    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2. * intersection + epsilon) / (mask_gt.sum() + mask_pred.sum() + epsilon)
    return np.clip(iou, 0, 1), np.clip(dice, 0, 1)  # 限制数值范围


def preprocess_image(image):
    """图像预处理增强"""
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced


def adaptive_binarize(image):
    """自适应阈值二值化"""
    # 使用Otsu算法自动选择阈值
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (thresh > 127).astype(np.uint8)


def process_images(label_dir, pred_dir):
    """改进后的处理流程"""
    iou_list, dice_list = [], []

    # 严格文件名匹配（去除后缀差异）
    label_files = sorted([f.split('.')[0] for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))])
    pred_files = sorted([f.split('.')[0] for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))])

    if set(label_files) != set(pred_files):
        missing = set(label_files) - set(pred_files)
        raise ValueError(f"预测图像缺失以下文件: {missing}")

    for file_id in label_files:
        # 读取图像（兼容不同后缀）
        label_path = os.path.join(label_dir, f"{file_id}.png")
        pred_path = os.path.join(pred_dir, f"{file_id}.png")

        # 读取并预处理
        label = np.array(Image.open(label_path).convert('L'))
        pred = np.array(Image.open(pred_path).convert('L'))

        # 预处理流程
        label = preprocess_image(label)
        pred = preprocess_image(pred)

        # 自适应二值化
        label_bin = adaptive_binarize(label)
        pred_bin = adaptive_binarize(pred)

        # 尺寸校验
        if label_bin.shape != pred_bin.shape:
            pred_bin = cv2.resize(pred_bin, label_bin.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # 计算指标
        iou, dice = calculate_iou_dice(label_bin, pred_bin)
        iou_list.append(iou)
        dice_list.append(dice)
        print(f"{file_id}: IoU={iou:.4f}, Dice={dice:.4f}")

    return np.array(iou_list), np.array(dice_list)


if __name__ == "__main__":
    label_dir = r"D:\UNet_py\Pytorch-UNet-master\Pytorch-UNet-master\unet\label"
    pred_dir = r"D:\UNet_py\Pytorch-UNet-master\Pytorch-UNet-master\unet\prediction"

    try:
        iou, dice = process_images(label_dir, pred_dir)

        # 异常值过滤（去除低于1个标准差的数据）
        iou_clean = iou[iou > (iou.mean() - iou.std())]
        dice_clean = dice[dice > (dice.mean() - dice.std())]

        print("\n=== 优化后统计结果 ===")
        print(f"IoU: {iou_clean.mean():.4f} ± {iou_clean.std():.4f} (原始: {iou.mean():.4f} ± {iou.std():.4f})")
        print(f"Dice: {dice_clean.mean():.4f} ± {dice_clean.std():.4f} (原始: {dice.mean():.4f} ± {dice.std():.4f})")

        # 保存详细结果
        np.savez("metrics.npz", iou=iou, dice=dice)

    except Exception as e:
        print(f"Error: {str(e)}")