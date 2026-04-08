"""
dice_score.py

功能: 指标计算函数
时间: 2026/03/20
版本: 1.0
"""

import torch
from torch import Tensor
from scipy.ndimage import distance_transform_edt


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """Average of Dice coefficient for all batches, or for a single mask"""
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
    """Average of Dice coefficient for all classes"""
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False) -> Tensor:
    """Dice loss (objective to minimize) between 0 and 1"""
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def focal_loss(input, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(input)
    loss = (
            - alpha * target * (1 - p) ** gamma * torch.log(p + 1e-8)
            - (1 - alpha) * (1 - target) * p ** gamma * torch.log(1 - p + 1e-8)
    )
    return loss.mean()


def tversky_loss(input, target, alpha=0.3, beta=0.7):
    smooth = 1e-6
    input = torch.sigmoid(input)
    TP = (input * target).sum()
    FP = ((1 - target) * input).sum()
    FN = (1 - input) * target.sum()
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky_index


def boundary_loss(input: Tensor, target: Tensor, device: torch.device = torch.device("cuda")) -> Tensor:
    # sigmoid 后得到概率
    input_prob = torch.sigmoid(input)

    batch_size = target.shape[0]
    loss = 0.0

    for b in range(batch_size):
        mask = target[b].cpu().numpy()
        # 计算前景和背景距离图
        pos_dist = distance_transform_edt(mask == 0)  # 前景边界到背景距离
        neg_dist = distance_transform_edt(mask == 1)  # 背景边界到前景距离
        dist_map = pos_dist + neg_dist
        dist_map = torch.tensor(dist_map, dtype=torch.float32, device=device)
        # loss = 预测概率 * 距离图
        loss += torch.mean(input_prob[b] * dist_map)

    return loss / batch_size
