"""
decoders.py

功能: 超参数解码器模块
时间: 2026/03/26
版本: 1.0
"""

from typing import Tuple, Dict, Any, cast

import numpy as np


class Decoder:
    """HLOCE 超参数解码器"""

    @classmethod
    def decode(cls, binary_seq: np.ndarray) -> Dict[str, Any]:
        """解码二进制编码"""
        seq = cast(Tuple, tuple(binary_seq.tolist()))
        hparams = {
            "blocks_number": cls.decode_blocks_number(seq[0:2]),   # 块数
            "filter_number": cls.decode_filter_number(seq[2:4]),   # 滤波器数量
            "filter_size": 3,                                      # 卷积核大小
            "activation": cls.decode_activation(seq[4:6]),         # 激活函数类型
            "pooling": seq[6],                                     # 池化层类型
            "optimizer_type": cls.decode_optimizer(seq[7:9]),      # 优化器类型
            "batch_size": cls.decode_batch_size(seq[9:11]),        # 批量大小
            "learning_rate": cls.bits_to_real(seq[11:21]),         # 学习率大小
            "use_batchnorm": seq[21],                              # 批归一化操作
            "use_dropout": cls.decode_dropout(seq[22:24])          # 随机丢弃操作
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
        result = min_value + (bit_value / 1023) * scale  # 映射到 [0.00001, 0.01]
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


class DecoderMixed:
    """HLOCE + CHLOCE 超参数解码器"""

    @classmethod
    def decode(cls, binary_seq: np.ndarray) -> Dict[str, Any]:
        """解码二进制编码"""
        seq = cast(Tuple, tuple(binary_seq.tolist()))
        hparams = {
            # 固定参数
            "blocks_number": 9,                                    # 块数
            "filter_size": 3,                                      # 卷积核大小

            # 网络结构参数
            "filters_number": cls.decode_filter_number(seq[0:2]),  # 滤波器数量

            # 网络模块类型
            "activation": cls.decode_activation(seq[2:4]),         # 激活函数类型
            "use_batchnorm": seq[4],                               # 批归一化
            "pooling": seq[5],                                     # 池化层类型

            # 训练超参数
            "batch_size": cls.decode_batch_size(seq[6:8]),         # 批量大小

            # 正则化参数
            "use_dropout": cls.decode_dropout(seq[8:10]),          # 随机丢弃

            # 优化器参数
            "optimizer_type": cls.decode_optimizer(seq[10:12]),    # 优化器类型

            # Attention 模块参数
            "attention_activation": seq[12],                       # 输出激活函数类型
            "attention_fusion": seq[13],                           # 融合方式
            "attention_depth": cls.decode_attention_up(seq[14:16]),   # Attention 启用深度
        }
        return hparams

    @staticmethod
    def decode_filter_number(key: Tuple[int, int]) -> int:
        """解码滤波器数量"""
        mapping = {
            (0, 0): 64,
            (0, 1): 64,
            (1, 0): 64,
            (1, 1): 64
        }
        return mapping[key]

    @staticmethod
    def decode_filter_size(key: Tuple[int, int]) -> int:
        """解码卷积核大小"""
        mapping = {
            (0, 0): 3,
            (0, 1): 5,
            (1, 0): 7,
            (1, 1): 3
        }
        return mapping[key]

    @staticmethod
    def decode_activation(key: Tuple[int, int]) -> int:
        """解码激活函数索引"""
        mapping = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        return mapping[key]

    @staticmethod
    def decode_optimizer(key: Tuple[int, int]) -> str:
        """解码优化器名称"""
        mapping = {
            (0, 0): "RMSprop",
            (0, 1): "Adam",
            (1, 0): "AdamW",
            (1, 1): "Adamax"
        }
        return mapping[key]

    @staticmethod
    def decode_batch_size(key: Tuple[int, int]) -> int:
        """解码批量大小"""
        mapping = {
            (0, 0): 4,
            (0, 1): 8,
            (1, 0): 16,
            (1, 1): 32
        }
        return mapping[key]

    @staticmethod
    def decode_dropout(key: Tuple[int, int]) -> int:
        """解码 dropout 类型索引"""
        mapping = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        return mapping.get(key)

    @staticmethod
    def decode_attention_up(key: Tuple[int, int]) -> int:
        """解码上采样模块索引"""
        mapping = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        return mapping.get(key)
