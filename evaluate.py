"""
evaluate.py

功能: 测试主函数
时间: 2026/03/20
版本: 1.0
"""

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.utils import test_single_volume


@torch.inference_mode()
def evaluate(
    net: torch.nn.Module, dataloader: DataLoader,
    device: torch.device, split: str, test_save_path: Optional[str] = None
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """对数据集进行推理并计算指标"""
    amp = False
    net.eval()

    MODE_DICT = {"val": "Validation", "test_vol": "Testing"}
    mode = MODE_DICT.get(split, "Unknown")

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        all_metrics = []  # 存储所有样本的指标
        for i_batch, sampled_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=mode):
            # h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
            metric_i = test_single_volume(
                image, label, net, split=split, classes=2, patch_size=[224, 224],
                test_save_path=test_save_path, case=case_name, z_spacing=1
            )
            all_metrics.append(np.array(metric_i))         # 将当前样本的指标添加到列表

        all_metrics = np.array(all_metrics)                # 转换为数组，形状为 (N, num_classes, 2)
        mean_metrics = np.mean(all_metrics, axis=0)        # 计算均值，形状 (num_classes, 2)
        std_metrics = np.std(all_metrics, axis=0, ddof=1)  # 计算标准差，使用样本标准差

        # 遍历每个类别输出结果
        for i in range(mean_metrics.shape[0]):
            logging.info(
                "Class %d - Mean Dice: %f ± %f, Mean iou: %f ± %f" %
                (i, mean_metrics[i][0], std_metrics[i][0], mean_metrics[i][1], std_metrics[i][1])
            )

        # 第一个类别
        dice, iou, acc, rec, pre = mean_metrics[0]
        dice_std, iou_std, acc_std, rec_std, pre_std = std_metrics[0]

        logging.info(
            "%s performance in best val model: mean_dice: %f ± %f, iou: %f ± %f" %
            (mode, dice, dice_std, iou, iou_std)
        )

    net.train()
    return dice, dice_std, iou, iou_std, acc, acc_std, rec, rec_std, pre, pre_std
