# traintest.py

import os
import gc
import random
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, cast

import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import evaluate
from unet import UNet3, UNet5, UNet7, UNet9
from utils.data_loading import Custom_dataset
from utils.dice_score import dice_loss
from npz_preprocess import RandomGenerator


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path("D:/unet_data/dataset_split/npzgood")
LIST_DIR = Path("D:/unet_data/dataset_split/dataset_split")


def worker_init_fn(worker_id: int) -> None:
    """初始化 DataLoader worker 的随机种子"""
    random.seed(1234 + worker_id)


def clear_gpu_memory(model: nn.Module, optimizer: optim.Optimizer, loss: nn.Module) -> None:
    """释放 GPU 显存"""
    model.cpu()                            # 将模型移到 CPU
    optimizer.zero_grad(set_to_none=True)  # 清除优化器状态
    del model                              # 删除模型
    del optimizer                          # 删除优化器
    del loss                               # 删除损失函数
    torch.cuda.empty_cache()               # 清除没有引用的缓存
    gc.collect()                           # 手动触发垃圾回收
    torch.cuda.synchronize()               # 同步 CUDA 流


def UNet(
        n_channels: int, n_classes: int, blocks_number: int, filter_number: int, filter_size: int,
        activation: int, pooling: int, use_dropout: int, use_batchnorm: int, use_attention: bool
) -> Optional[nn.Module]:
    """根据块数返回对应的 UNet 网络实例"""
    if blocks_number == 3:
        return UNet3(
            n_channels, n_classes, filter_number, filter_size, activation,
            pooling, use_dropout, use_batchnorm, bilinear=False, use_attention=use_attention
        )
    if blocks_number == 5:
        return UNet5(
            n_channels, n_classes, filter_number, filter_size, activation,
            pooling, use_dropout, use_batchnorm, bilinear=False, use_attention=use_attention
        )
    if blocks_number == 7:
        return UNet7(
            n_channels, n_classes, filter_number, filter_size, activation,
            pooling, use_dropout, use_batchnorm, bilinear=False, use_attention=use_attention
        )
    if blocks_number == 9:
        return UNet9(
            n_channels, n_classes, filter_number, filter_size, activation,
            pooling, use_dropout, use_batchnorm, bilinear=False, use_attention=use_attention
        )

    print("block number error")
    return None


class Decoder:
    """超参数解码器"""

    @classmethod
    def decode(cls, binary_seq: np.ndarray) -> Dict[str, Any]:
        """解码二进制编码"""
        seq = cast(Tuple, tuple(binary_seq.tolist()))
        hparams = {
            "blocks_number": cls.decode_blocks_number(seq[0:2]),
            "filter_number": cls.decode_filter_number(seq[2:4]),
            "filter_size": 3,
            "activation": cls.decode_activation(seq[4:6]),
            "pooling": seq[6],
            "optimizer_type": cls.decode_optimizer(seq[7:9]),
            "batch_size": cls.decode_batch_size(seq[9:11]),
            "learning_rate": cls.bits_to_real(seq[11:21]),
            "use_batchnorm": seq[21],
            "use_dropout": cls.decode_dropout(seq[22:24])
        }
        return hparams

    @staticmethod
    def decode_blocks_number(key: Tuple[int, int]) -> int:
        """将二进制序列解码为块数"""
        mapping = {
            (0, 0): 9,
            (0, 1): 9,
            (1, 0): 9,
            (1, 1): 9
        }
        return mapping[key]

    @staticmethod
    def decode_filter_number(key: Tuple[int, int]) -> int:
        """将二进制序列解码为滤波器数量"""
        mapping = {
            (0, 0): 4,
            (0, 1): 8,
            (1, 0): 16,
            (1, 1): 32
        }
        return mapping[key]

    @staticmethod
    def decode_filter_size(key: Tuple[int, int]) -> int:
        """将二进制序列解码为卷积核大小"""
        mapping = {
            (0, 0): 3,
            (0, 1): 5,
            (1, 0): 7,
            (1, 1): 3
        }
        return mapping[key]

    @staticmethod
    def decode_activation(key: Tuple[int, int]) -> int:
        """将二进制序列解码为激活函数索引"""
        mapping = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        return mapping[key]

    @staticmethod
    def decode_pooling(key: int) -> int:
        """将二进制值解码为池化类型索引"""
        return key

    @staticmethod
    def decode_optimizer(key: Tuple[int, int]) -> str:
        """将二进制序列解码为优化器名称"""
        mapping = {
            (0, 0): "Adamax",   # SGD
            (0, 1): "RMSprop",
            (1, 0): "Adam",
            (1, 1): "AdamW"
        }
        return mapping[key]

    @staticmethod
    def decode_batch_size(key: Tuple[int, int]) -> int:
        """将二进制序列解码为批量大小"""
        mapping = {
            (0, 0): 4,
            (0, 1): 8,
            (1, 0): 16,
            (1, 1): 32
        }
        return mapping[key]

    # 小批量训练带来了更多权重更新的可能性，在学习率较大时难以收敛
    # 大批量训练则更接近用整个数据集进行训练，只要数据集正确，则更容易收敛，则可以用较大的学习率也不用担心收敛问题
    @staticmethod
    def bits_to_real(bit_sequence: Tuple[int, int, int, int, int, int, int, int, int, int]) -> float:
        """根据 10 比特的输入比特序列，将其映射到 [0.00001, 0.01] 范围内的实数"""
        bit_string = "".join(map(str, bit_sequence))     # 转换为二进制字符串
        bit_value = int(bit_string, 2)                   # 按二进制解析为整数
        max_value = 0.001                                # 映射范围的最大值
        min_value = 0.00001                              # 映射范围的最小值
        scale = max_value - min_value                    # 映射范围
        result = max_value - (bit_value / 1023) * scale  # 映射到 [0.00001, 0.01]
        return result

    @staticmethod
    def decode_dropout(key: Tuple[int, int]) -> int:
        """将二进制序列解码为 dropout 索引"""
        mapping = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        return mapping.get(key, 0)


def testFunction(params_list: List[np.ndarray], use_attention: bool = False) -> List[float]:
    """根据参数列表训练模型，返回每组参数对应的损失值"""
    losses = []

    for params in params_list:
        if len(params) == 22:
            params = np.r_[1, 1, params]

        # 从参数中提取不同的超参数
        hparams = Decoder.decode(params)
        hparams["use_attention"] = use_attention
        logging.info(f"二进制超参数解码结果: {hparams.values()}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {device}")

        # Change here to adapt to your data
        # n_channels = 3 for RGB image
        # n_classes is the number of probabilities you want to get per pixel
        model = UNet(
            3, 1,
            hparams["blocks_number"], hparams["filter_number"], hparams["filter_size"],
            hparams["activation"], hparams["pooling"],
            hparams["use_dropout"], hparams["use_batchnorm"], hparams["use_attention"]
        )
        model = model.to(device=device)

        logging.info(
            f"Network:\n"
            f"\t\t{model.n_channels} input channels\n"
            f"\t\t{model.n_classes} output channels (classes)\n"
            f"\t\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
        )

        try:
            loss = train_model(
                model=model,
                device=device,
                epochs=60,
                params=list(params),
                hparams=hparams
            )

        except torch.cuda.OutOfMemoryError:
            logging.error(
                "Detected OutOfMemoryError! "
                "Enabling checkpointing to reduce memory usage, but this slows down training. "
                "Consider enabling AMP (--amp) for fast and memory efficient training. "
            )
            torch.cuda.empty_cache()
            model.use_checkpointing()
            loss = train_model(
                model=model,
                device=device,
                epochs=60,
                params=list(params),
                hparams=hparams
            )

        try:
            losses.append(loss)
        except AttributeError:
            print(loss)

    return losses  # 1 - dice 作为适应度值返回


def train_model(
        model: nn.Module,
        device: torch.device,
        epochs: int,
        params: List[int],
        hparams: Dict[str, Any],
        img_scale: float = 0.5,
        amp: bool = False,
        save_checkpoint: bool = True,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0
) -> float:
    # 提取超参数
    blocks_number = hparams["blocks_number"]
    filter_number = hparams["filter_number"]
    filter_size = hparams["filter_size"]
    activation = hparams["activation"]
    pooling = hparams["pooling"]
    optimizer_type = hparams["optimizer_type"]
    batch_size = hparams["batch_size"]
    learning_rate = hparams["learning_rate"]
    use_batchnorm = hparams["use_batchnorm"]
    use_dropout = hparams["use_dropout"]
    use_attention = hparams["use_attention"]

    """使用指定超参数训练 U-Net 模型，并返回 1 - dice 作为适应度值"""
    # 1. 创建数据集
    try:
        # BASE_DIR 是所有 npz 文件存放的路径，LIST_DIR 是记录挑选进训练集（train）的病例的 txt 文件
        train_set = Custom_dataset(
            BASE_DIR, LIST_DIR, split="train",
            transform=transforms.Compose([RandomGenerator(output_size=[224, 224])])
        )
        val_set = Custom_dataset(BASE_DIR, LIST_DIR, split="val", transform=None)

    except (AssertionError, RuntimeError, IndexError):
        print(f"Data loading error! ")
        return float("inf")

    # 2. 划分训练集和验证集
    n_train = len(train_set)
    n_val = len(val_set)

    # 3. 训练和验证数据加载器
    loader_args = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        worker_init_fn=worker_init_fn, **loader_args
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        drop_last=True, **loader_args
    )

    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Blocks Number:   {blocks_number}
        Filter Number:   {filter_number}
        Filter Size:     {filter_size}
        Activation:      {activation}
        Pooling:         {pooling}
        Optimizer Type:  {optimizer_type}
        Batch Size:      {batch_size}
        Learning Rate:   {learning_rate}
        Use Batchnorm:   {use_batchnorm}
        Use Dropout:     {use_dropout}
        Use Attention:   {use_attention}
        Training Size:   {n_train}
        Validation Size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images Scaling:  {img_scale}
        Mixed Precision: {amp}
    """)

    # 4. 设置优化器、损失函数、学习率调度器以及 AMP 的损失缩放
    if optimizer_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
            momentum=momentum, nesterov=True
        )
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
            momentum=momentum
        )
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. 开始训练
    iter_num = 0
    max_iterations = epochs * len(train_loader)
    low_accuracy_count = 0
    val_score = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["label"]

                assert images.shape[1] == model.n_channels, (
                    f"Network has been defined with {model.n_channels} input channels, "
                    f"but loaded image have {images.shape[1]} channels. Please check that "
                    f"the image are loaded correctly. "
                )

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # 自动混合精度训练 二元交叉熵 dice 混合
                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # 使用梯度缩放器进行反向传播和梯度计算
                # 对梯度进行裁剪，防止梯度爆炸
                # 更新优化器参数，并更新梯度缩放器的缩放值
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                lr_ = learning_rate * (1.0 - iter_num / max_iterations)**0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
                iter_num = iter_num + 1

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})

        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace("/", ".")
            if not (torch.isinf(value) | torch.isnan(value)).any():
                histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

        # 验证阶段
        val_score = evaluate(model, val_loader, device, split="val")
        logging.info(f"Validation Dice score: {val_score}")

        if epoch >= 20:
            if val_score < 0.65:         # 如果准确率低于 0.65，增加计数器
                low_accuracy_count += 1
            else:                        # 如果准确率高于 0.65，重置计数器
                low_accuracy_count = 0

            if low_accuracy_count >= 3:  # 如果连续三轮都低于 0.65，跳出循环
                logging.info(f"Stopping early at epoch {epoch} due to low accuracy. ")
                break

    # 保存分割效果较好的模型
    if val_score >= 0.88:
        save_dir = "good_model"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存模型
        model_name = f"model_dice_{val_score:.4f},params_{params}.pth"
        save_path = os.path.join(save_dir, model_name)

        checkpoint = {
            "model_name": model.__class__.__name__,
            "model_state": model.state_dict(),
            "params": params,
            "blocks_number": blocks_number,
            "filters_number": filter_number,
            "filter_size": filter_size,
            "activation": activation,
            "pooling": pooling,
            "optimizer_type": optimizer_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_batchnorm": use_batchnorm,
            "use_dropout": use_dropout,
            "use_attention": use_attention
        }

        torch.save(checkpoint, save_path)
        print(f"Model has saved, val_sccore = {val_score}, params = {params}")

    clear_gpu_memory(model, optimizer, criterion)
    return 1 - val_score


# 用到测试集时删除注释符号使用
if __name__ == "__main__":
    base_path = "D:/Pytorch-UNet-master/good_model/"
    saved_models = os.path.join(
        base_path,
        "model_dice_0.8857,params_[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1].pth"
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    test_save_dir = "D:/unet_test"

    if saved_models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型，提取超参数
        checkpoint = torch.load(saved_models, map_location=device)
        params = checkpoint["params"]

        model = UNet(
            n_channels=3,
            n_classes=1,
            blocks_number=checkpoint["blocks_number"],
            filter_number=checkpoint["filters_number"],
            filter_size=checkpoint["filter_size"],
            activation=checkpoint["activation"],
            pooling=checkpoint["pooling"],
            use_dropout=checkpoint["use_dropout"],
            use_batchnorm=checkpoint["use_batchnorm"],
            use_attention=checkpoint["use_attention"]
        )

        model.load_state_dict(checkpoint["model_state"])

        print(f"Loaded model: {os.path.basename(saved_models)}")
        model = model.to(device)

        # 开始对训练完的模型进行测试
        split = "test_vol"
        db_test = Custom_dataset(BASE_DIR, LIST_DIR, split=split)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        evaluate(model, testloader, device, split, test_save_path=test_save_dir)
        torch.cuda.empty_cache()


# if __name__ == "__main__":
#     test_list = [
#         np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1])
#     ]
#     testFunction(test_list, use_attention=True)
