# evaluate.py

import logging
from typing import Optional

import torch
from tqdm import tqdm
import numpy as np

from utils.utils import test_single_volume


@torch.inference_mode()
def evaluate(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
    test_save_path: Optional[str] = None
) -> float:
    """对整个数据集进行推理并计算平均 Dice 指标"""
    amp = False
    net.eval()

    mode_dict = {"val": "Validation", "test_vol": "Testing"}
    mode = mode_dict.get(split, "Unknown")

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        all_metrics = []  # 存储所有样本的指标
        for i_batch, sampled_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=mode):
            # h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
            metric_i = test_single_volume(image, label, net, split=split, classes=2, patch_size=[224, 224],
                                          test_save_path=test_save_path, case=case_name, z_spacing=1)
            all_metrics.append(np.array(metric_i))         # 将当前样本的指标添加到列表

        all_metrics = np.array(all_metrics)                # 转换为数组，形状为 (N, num_classes, 2)
        mean_metrics = np.mean(all_metrics, axis=0)        # 计算均值，形状 (num_classes, 2)
        std_metrics = np.std(all_metrics, axis=0, ddof=1)  # 计算标准差，使用样本标准差

        # 遍历每个类别输出结果
        for i in range(mean_metrics.shape[0]):
            logging.info("Class %d - Mean Dice: %f ± %f, Mean iou: %f ± %f" %
                         (i, mean_metrics[i][0], std_metrics[i][0], mean_metrics[i][1], std_metrics[i][1]))

        # 假设关注第一个类别（根据实际情况调整索引）
        performance = mean_metrics[0][0]
        performance_std = std_metrics[0][0]
        iou = mean_metrics[0][1]
        iou_std = std_metrics[0][1]

        logging.info("%s performance in best val model: mean_dice: %f ± %f, iou: %f ± %f" %
                     (mode, performance, performance_std, iou, iou_std))

    net.train()
    return performance
