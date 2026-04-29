"""
HLOCE_v2_0.py

功能: 使用 HLOCE 对 U-Net 超参数调优
时间: 2025/12/19
版本: 2.0
修改: 在 1.0 版本上封装了 HLOCE 迭代，去掉了对网络层数的改进
"""

import os
import time
import logging
from typing import Optional, Any, Union

import numpy as np

from traintest import testFunction


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class HLOCEOptimizer:
    """HLOCE 参数和迭代类"""

    def __init__(self, pop_size: int, bit: int):
        self.pop_size = pop_size
        self.bit = bit

        # 伯努利参数初始化
        self.ber_params_before = self._initialize_multinoulli_parameters(bit)

        # 基本参数
        self.a = 0.83    # 交叉熵参数

        self.pr = np.zeros(pop_size)  # 随机学习概率
        self.pi = np.zeros(pop_size)  # 个体学习概率
        self.ps = np.zeros(pop_size)  # 决定交叉熵学习还是社会学习

        self.Kr, self.prMax = 2, 0.1                   # 计算 pr[i]
        self.K1, self.Ki, self.piMax = 0.83, 4, 0.92   # 计算 pi[i]
        self.K2, self.Ks, self.psMax = 0.82, 3, 0.87   # 计算 ps[i]

        self.pr0, self.pi0, self.ps0 = 0.005, 0.88, 0.50  # sum 为 0 时的取值

    def update_population(
            self, popus: np.ndarray, IKD: np.ndarray,
            SKD: np.ndarray, IKDfits: np.ndarray
    ) -> np.ndarray:
        """HLOCE 迭代"""
        # 计算交叉熵概率
        ber_params = self._ce_prob(IKDfits, IKD, 3)

        # 平滑更新概率参数
        ber_params_after = self.a * ber_params + (1 - self.a) * self.ber_params_before
        self.ber_params_before = ber_params_after.copy()

        # 更新种群
        for i in range(self.pop_size):
            # 计算当前 i 对应的 pr[i], pi[i], ps[i]
            total = np.sum(np.abs(IKD[i] - SKD))

            if total == 0:     # total 为 0 时，使用基本 HLO 的 pr 和 pi
                self.pr[i] = self.pr0
                self.pi[i] = self.pi0
                self.ps[i] = self.ps0
            else:              # total 不为 0 时，使用有效信息的计算方法
                self.pr[i] = min(self.Kr / total, self.prMax)             # 控制 pr[i] 的上限
                self.pi[i] = min(self.K1 + self.Ki / total, self.piMax)   # 控制 pi[i] 的上限
                self.ps[i] = min(self.K2 + self.Ks / total, self.psMax)   # 控制 ps[i] 的上限

            for j in range(self.bit):
                prob1 = np.random.uniform()
                prob2 = np.random.uniform()
                if prob1 < self.pr[i]:      # 随机学习
                    popus[i][j] = np.random.randint(0, 2)
                elif prob1 < self.pi[i]:    # 个体学习
                    popus[i][j] = IKD[i][j]
                else:
                    if prob2 < self.ps[i]:  # 交叉熵学习
                        popus[i][j] = 1 if np.random.rand() < ber_params_after[j] else 0
                    else:                   # 社会学习
                        popus[i][j] = SKD[j]

        return popus

    @staticmethod
    def _initialize_multinoulli_parameters(m: int) -> np.ndarray:
        """初始化伯努利参数"""
        return np.random.rand(m)

    @staticmethod
    def _ce_prob(IKDfits: np.ndarray, IKD: np.ndarray, Ne: int) -> np.ndarray:
        """使用多元伯努利分布计算交叉熵概率"""
        indices = np.argpartition(IKDfits, Ne)[:Ne]        # 找到适应度最小的 Ne 个个体索引
        elite_population = IKD[indices]                    # 提取精英个体
        probabilities = np.mean(elite_population, axis=0)  # 计算每个基因位点为 1 的概率
        return probabilities


def HLOCE_v2_0(
        max_iter: int = 10,
        pop_size: int = 10,
        bit: int = 22,
        rl: int = 50,
        use_attention: bool = True
) -> Optional[list[Union[np.ndarray, np.int64]]]:
    """HLOCE 优化二进制超参数"""
    # 记录程序开始时间
    start_time = time.time()

    # 创建 HLOCE 优化器，用于二进制超参数
    HLOCE_optimiter = HLOCEOptimizer(pop_size, bit)

    # 创建文件，保存最优解及运行时间
    output_file = f"HLOCE_test_v2_0_results_{'attention' if use_attention else 'baseline'}.txt"
    with open(output_file, "w") as file:
        file.write("HLOCE 优化过程结果：\n")

    # 初始化种群
    pop: dict[str, Any] = {
        "popus": np.random.randint(0, 2, (pop_size, bit)),
        "fitness": None
    }

    pop["fitness"] = testFunction(pop["popus"], use_attention=use_attention)  # 不同参数模型的损失
    ind = np.argmin(pop["fitness"])                                           # 最小值的位置

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
    count = np.zeros(pop_size)
    parameters = None

    # HLOCE 迭代
    for it in range(max_iter):
        # HLOCE 更新二进制参数
        pop["popus"] = HLOCE_optimiter.update_population(
            pop["popus"], individual["IKD"], global_best["SKD"], individual["IKDfits"]
        )

        # 更新适应度值
        pop["fitness"] = testFunction(pop["popus"], use_attention=use_attention)

        # 更新个体最优
        for i in range(0, pop_size):
            if pop["fitness"][i] < individual["IKDfits"][i]:
                individual["IKDfits"][i] = pop["fitness"][i]
                individual["IKD"][i] = pop["popus"][i].copy()
                count[i] = 0
            else:
                count[i] += 1

            # 重新初始化
            if count[i] == rl:
                individual["IKD"][i] = np.random.randint(0, 2, bit)
                individual["IKDfits"][i] = testFunction(
                    params_list=[individual["IKD"][i]],
                    use_attention=use_attention
                )[0]
                count[i] = 0

        # 寻找全局最优
        ind = np.argmin(individual["IKDfits"])
        if individual["IKDfits"][ind] < global_best["SKDfit"]:
            global_best["SKDfit"] = individual["IKDfits"][ind]
            global_best["SKD"] = individual["IKD"][ind].copy()

            # 记录当前最优解
            parameters = [
                global_best["SKD"][0:2],    # 滤波器数量 4, 8, 16, 32
                global_best["SKD"][2:4],    # 激活函数 ReLU, ELU, LeakyReLU, RReLU
                global_best["SKD"][4],      # 池化层 max mean
                global_best["SKD"][5:7],    # 优化器 Adamax, RMSprop, Adam, AdamW
                global_best["SKD"][7:9],    # 批次大小 4, 8, 16, 32
                global_best["SKD"][9:19],   # 学习率 [0.00001, 0.001]
                global_best["SKD"][19],     # 批量归一化
                global_best["SKD"][20:22]   # dropout
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
    best_params = HLOCE_v2_0()
    logging.info("Best hyperparameters found:")
    logging.info(best_params)
