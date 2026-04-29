"""
HLOCE_v1_0.py

功能: 使用 HLOCE 对 U-Net 超参数调优
时间: 2025/11/30
版本: 1.0
"""

import os
import time
import logging
from typing import Optional, Any, Union

import numpy as np

from traintest import testFunction


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fix_invalid_combinations(popus: np.ndarray) -> np.ndarray:
    """如果个体是 (blocks = 7 或 9) 且 filters = 32，则把 filters 改为 16，以免耗时过长"""
    popus_fixed = popus.copy()

    for i in range(len(popus_fixed)):
        block_bits = tuple(popus_fixed[i, 0:2])
        filter_bits = tuple(popus_fixed[i, 2:4])

        if block_bits in [(1, 0), (1, 1)]:  # blocks = 7 或 9
            if filter_bits == (1, 1):       # filters = 32
                popus_fixed[i, 3] = 0       # 改为 filters = 16 -> (1, 0)

    return popus_fixed


def initialize_multinoulli_parameters(m: int) -> np.ndarray:
    """初始化伯努利参数"""
    return np.random.rand(m)


def ce_prob(IKDfits: np.ndarray, IKD: np.ndarray, Ne: int) -> np.ndarray:
    """使用多元伯努利分布计算交叉熵概率"""
    indices = np.argpartition(IKDfits, Ne)[:Ne]        # 找到适应度最小的 Ne 个个体索引
    elite_population = IKD[indices]                    # 提取精英个体
    probabilities = np.mean(elite_population, axis=0)  # 计算每个基因位点为 1 的概率
    return probabilities


def HLOCE_v1_0(
        maxIter: int = 10,
        popSize: int = 10,
        bit: int = 24,
        rl: int = 50
) -> Optional[list[Union[np.ndarray, np.int64]]]:
    """HLOCE 优化超参数"""
    # 记录程序开始时间
    start_time = time.time()

    # 基本参数设置
    a = 0.8    # 交叉熵参数

    pr = np.full(popSize, fill_value=0.0)   # 随机学习概率
    pi = np.full(popSize, fill_value=0.0)   # 个体学习概率
    ps = np.full(popSize, fill_value=0.0)   # 决定交叉熵学习还是社会学习

    # 计算 pr[i]
    Kr = 2
    prMax = 0.1     # pr[i] 的上限

    # 计算 pi[i]
    K1 = 0.83
    Ki = 5
    piMax = 0.9     # pi[i] 的上限

    # 计算 ps[i]
    K2 = 0.63
    Ks = 3
    psMax = 0.67    # ps[i] 的上限

    # 有效信息相关参数
    lenVar = bit
    pr0 = 0.005    # 在 sum 为 0 时使用的 pr 数值
    pi0 = 0.83     # 在 sum 为 0 时使用的 pi 数值
    ps0 = 0.64     # 在 sum 为 0 时使用的 ps 数值

    # 创建输出文件
    output_file = "HLOCE_test_v1_0_results.txt"
    with open(output_file, "w") as file:
        file.write("HLOCE优化过程结果：\n")

    # 初始化种群
    pop: dict[str, Any] = {
        "popus": np.random.randint(0, 2, (popSize, lenVar)),
        "fitness": None
    }

    pop["popus"] = fix_invalid_combinations(pop["popus"])
    pop["fitness"] = testFunction(pop["popus"])  # 不同参数模型的损失
    ind = np.argmin(pop["fitness"])              # 最小值的位置

    # 初始化局部最优
    individual: dict[str, Any] = {
        "IKD": pop["popus"].copy(),
        "IKDfits": pop["fitness"].copy(),
    }

    # 初始化全局最优
    global_best: dict[str, Any] = {
        "SKD": pop["popus"][ind].copy(),
        "SKDfit": pop["fitness"][ind]
    }

    # 初始化计数器
    count = np.zeros(popSize)

    # 初始化交叉熵参数
    ber_params_before = initialize_multinoulli_parameters(lenVar)

    # HLOCE 迭代
    parameters = None
    for it in range(maxIter):
        # 计算交叉熵概率
        ber_params = ce_prob(individual["IKDfits"], individual["IKD"], 5)

        # 平滑更新概率参数
        ber_params_after = a * ber_params + (1 - a) * ber_params_before
        ber_params_before = ber_params_after.copy()

        # 更新种群
        for i in range(popSize):
            # 首先计算当前 i 对应的 pr[i] 和 pi[i]
            total = 0
            for j in range(lenVar):
                total += abs(individual["IKD"][i][j] - global_best["SKD"][j])

            if total == 0:     # total 为 0 时，使用基本 HLO 的 pr 和 pi
                pr[i] = pr0
                pi[i] = pi0
                ps[i] = ps0
            else:              # total 不为 0 时，使用有效信息的计算方法
                pr[i] = min(Kr / total, prMax)        # 控制 pr[i] 的上限
                pi[i] = min(K1 + Ki / total, piMax)   # 控制 pi[i] 的上限
                ps[i] = min(K2 + Ks / total, psMax)   # 控制 ps[i] 的上限

            for j in range(lenVar):
                prob1 = np.random.uniform()
                prob2 = np.random.uniform()
                if prob1 < pr[i]:      # 随机学习
                    pop["popus"][i][j] = np.random.randint(low=0, high=2)
                elif prob1 < pi[i]:    # 个体学习
                    pop["popus"][i][j] = individual["IKD"][i][j]
                else:
                    if prob2 < ps[i]:  # 交叉熵学习
                        pop["popus"][i][j] = 1 if np.random.rand() < ber_params_after[j] else 0
                    else:              # 社会学习
                        pop["popus"][i][j] = global_best["SKD"][j]

        # 更新适应度值
        pop["fitness"] = testFunction(pop["popus"])

        # 更新个体最优
        for i in range(0, popSize):
            if pop["fitness"][i] < individual["IKDfits"][i]:
                individual["IKDfits"][i] = pop["fitness"][i]
                individual["IKD"][i] = pop["popus"][i].copy()
                count[i] = 0
            else:
                count[i] += 1

            # 重新初始化
            if count[i] == rl:
                candidate = np.random.randint(0, 2, lenVar)
                candidate = fix_invalid_combinations(candidate.reshape(1, -1))[0]
                individual["IKD"][i] = candidate
                individual["IKDfits"][i] = testFunction(individual["IKD"][i].reshape(1, -1))[0]
                count[i] = 0

        # 寻找全局最优
        ind = np.argmin(individual["IKDfits"])
        if individual["IKDfits"][ind] < global_best["SKDfit"]:
            global_best["SKDfit"] = individual["IKDfits"][ind]
            global_best["SKD"] = individual["IKD"][ind].copy()

            # 记录当前最优解
            parameters = [
                global_best["SKD"][0:2],    # 块数 3, 5, 7, 9
                global_best["SKD"][2:4],    # 滤波器数量 4, 8, 16, 32
                global_best["SKD"][4:6],    # 激活函数 ReLU, ELU, LeakyReLU, RReLU
                global_best["SKD"][6],      # 池化层 max mean
                global_best["SKD"][7:9],    # 优化器 Adamax, RMSprop, Adam, AdamW
                global_best["SKD"][9:11],   # 批次大小 4, 8, 16, 32
                global_best["SKD"][11:21],  # 学习率 [0.00001, 0.001]
                global_best["SKD"][21],     # 批量归一化
                global_best["SKD"][22:24]   # dropout
            ]

            # 将参数和适应度值写入文件
            with open(output_file, "a") as file:
                file.write(f"第{it + 1}代：\n")
                file.write(f"参数：{parameters}\n")
                file.write(f"适应度值：{global_best['SKDfit']}\n\n")

        logging.info(f"第{it}代结果：{global_best['SKDfit']}")

    # 记录总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    with open(output_file, "a") as file:
        file.write(f"总运行时间：{total_time:.2f}秒\n")

    # 返回最优权值结果
    return parameters


if __name__ == "__main__":
    best_params = HLOCE_v1_0()
    logging.info("Best hyperparameters found:")
    logging.info(best_params)
