"""
HLOCE_test_v3_0.py

功能: 使用 HLOCE 对 U-Net 超参数调优
时间: 2026/03/26
版本: 3.0
修改: 在 2.0 版本上增加了 CHLOCE 优化连续超参数
"""

import os
import time
import logging
from typing import Optional, List, Any, Union

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


class CHLOCEOptimizer:
    """CHLOCE 参数和迭代类"""

    def __init__(self, pop_size: int, dim: int, xMax: float, xMin: float):
        self.pop_size = pop_size
        self.dim = dim
        self.xMax = xMax
        self.xMin = xMin

        # 高斯参数初始化
        self.gaussian_params_before = self._initialize_gaussian_parameters(dim)

        # 基本参数
        self.a = 0.8    # 交叉熵参数

        self.K1, self.K2, self.K3 = 0.1, 0.8, 0.4

        self.pi = np.zeros(self.pop_size)

        self.pr = 0.005
        self.K, self.Ki = 0.81, 0.2
        self.ps = 0.64

        self.pi0 = 0.8

    def update_population(
            self, popus: np.ndarray, IKD: np.ndarray,
            SKD: np.ndarray, IKDfits: np.ndarray
    ) -> np.ndarray:
        """CHLOCE 迭代"""
        # 交叉熵高斯参数更新
        gaussianParams = self._ce_gaussian(IKD, IKDfits, 3)
        gaussianParams_after = []

        for j in range(self.dim):
            current = gaussianParams[j]
            previous = self.gaussian_params_before[j]

            mean = self.a * current[0] + (1 - self.a) * previous[0]
            std = self.a * current[1] + (1 - self.a) * previous[1]
            gaussianParams_after.append([mean, std])

        self.gaussian_params_before = [gp.copy() for gp in gaussianParams_after]

        # 计算差异度
        diff = np.abs(IKD - SKD)
        maxDifference = np.max(diff, axis=0)
        maxDifference[maxDifference < 1e-12] = 1e-12

        # 更新种群
        for i in range(self.pop_size):
            normalized = diff[i] / maxDifference
            sum_diff = np.sqrt(np.sum(normalized ** 2))

            # 计算 pi[i]
            if sum_diff == 0:
                self.pi[i] = self.pi0
            else:
                self.pi[i] = self.K + self.Ki * sum_diff / self.dim

            # 更新种群每个维度
            for j in range(self.dim):
                prob = np.random.rand()
                if prob < self.pr:                     # 随机学习
                    popus[i][j] = self.xMin + np.random.rand() * (self.xMax - self.xMin)
                elif prob < self.pi[i]:                # 个体学习
                    popus[i][j] = np.random.normal(IKD[i][j], self.K1 * abs(SKD[j] - IKD[i][j]))
                else:
                    if np.random.rand() < self.ps:     # 交叉熵学习
                        mean, std = gaussianParams_after[j]
                        popus[i][j] = np.random.normal(mean, std)
                    else:                              # 社会学习
                        direction = SKD[j] - IKD[i][j]
                        sign = 1 if np.random.rand() < 0.5 else -1
                        popus[i][j] = (
                                sign * self.K2 * np.random.rand() * direction
                                + np.random.normal(SKD[j], self.K3 * abs(direction))
                        )

                popus[i][j] = np.clip(popus[i][j], self.xMin, self.xMax)

        return popus

    @staticmethod
    def _initialize_gaussian_parameters(dim: int) -> List[List[float]]:
        """初始化高斯参数"""
        return [[np.random.rand(), np.random.rand()] for _ in range(dim)]

    @staticmethod
    def _ce_gaussian(IKD: np.ndarray, IKDfits: np.ndarray, Ne: int) -> List[List[float]]:
        """计算交叉熵高斯参数"""
        indices = np.argpartition(IKDfits, Ne)[:Ne]
        elite = IKD[indices]
        mean = np.mean(elite, axis=0)
        std = np.std(elite, axis=0)
        return [[m, s] for m, s in zip(mean, std)]


def HLOCE_v3_0(
        max_iter: int = 10,
        pop_size: int = 10,
        bit: int = 12,
        lr_dim: int = 1,
        rl: int = 50,
        use_attention: bool = True
) -> Optional[list[Union[np.ndarray, np.int64]]]:
    """HLOCE 优化二进制超参数，CHLOCE 优化学习率"""
    # 记录程序开始时间
    start_time = time.time()

    # 创建 HLOCE 优化器，用于二进制超参数
    HLOCE_optimiter = HLOCEOptimizer(pop_size, bit)

    # 创建 CHLOCE 优化器，用于学习率
    CHLOCE_optimiter = CHLOCEOptimizer(pop_size, lr_dim, xMax=0.001, xMin=0.00001)

    # 创建输出文件
    output_file = f"HLOCE_test_v3_0_results_{'attention' if use_attention else 'baseline'}.txt"
    with open(output_file, "w") as file:
        file.write("HLOCE + CHLOCE 优化过程结果：\n")

    # 初始化 HLOCE 种群
    pop: dict[str, Any] = {
        "popus": np.random.randint(0, 2, (pop_size, bit)),
        "fitness": None
    }

    # 初始化 CHLOCE 种群
    lr_pop = np.random.uniform(CHLOCE_optimiter.xMin, CHLOCE_optimiter.xMax, (pop_size, lr_dim))

    # 调用 testFunction，初始适应度
    pop["fitness"] = testFunction(pop["popus"], lr_pop, use_attention=use_attention)
    ind = np.argmin(pop["fitness"])

    # 初始化局部最优
    individual: dict[str, Any] = {
        "IKD": pop["popus"].copy(),
        "IKDfits": pop["fitness"].copy()
    }
    lr_individual: dict[str, Any] = {
        "IKD": lr_pop.copy()
    }

    # 初始化全局最优
    global_best: dict[str, Any] = {
        "SKD": pop["popus"][ind].copy(),
        "SKDfit": pop["fitness"][ind],
        "lr_SKD": lr_pop[ind].copy()
    }

    # 初始化计数器
    count = np.zeros(pop_size)
    parameters = None

    # HLOCE + CHLOCE 迭代
    for it in range(max_iter):
        # HLOCE 更新二进制参数
        pop["popus"] = HLOCE_optimiter.update_population(
            pop["popus"], individual["IKD"], global_best["SKD"], individual["IKDfits"]
        )

        # CHLOCE 更新学习率
        lr_pop = CHLOCE_optimiter.update_population(
            lr_pop, lr_individual["IKD"], global_best["lr_SKD"], individual["IKDfits"]
        )

        # 调用 testFunction，更新适应度
        pop["fitness"] = testFunction(pop["popus"], lr_pop, use_attention=use_attention)

        # 更新个体最优
        for i in range(pop_size):
            if pop["fitness"][i] < individual["IKDfits"][i]:
                individual["IKDfits"][i] = pop["fitness"][i]
                individual["IKD"][i] = pop["popus"][i].copy()
                lr_individual["IKD"][i] = lr_pop[i].copy()
                count[i] = 0
            else:
                count[i] += 1

            # 重新初始化
            if count[i] == rl:
                individual["IKD"][i] = np.random.randint(0, 2, bit)
                lr_individual["IKD"][i] = np.random.uniform(CHLOCE_optimiter.xMin, CHLOCE_optimiter.xMax, lr_dim)
                individual["IKDfits"][i] = testFunction(
                    params_list=[individual["IKD"][i]],
                    lr_pop=[lr_individual["IKD"][i]],
                    use_attention=use_attention
                )[0]
                count[i] = 0

        # 寻找全局最优
        ind = np.argmin(individual["IKDfits"])
        if individual["IKDfits"][ind] < global_best["SKDfit"]:
            global_best["SKDfit"] = individual["IKDfits"][ind]
            global_best["SKD"] = individual["IKD"][ind].copy()
            global_best["lr_SKD"] = lr_individual["IKD"][ind].copy()

            parameters = [
                global_best["SKD"][0:2],    # 滤波器数量 4, 8, 16, 32
                global_best["SKD"][2:4],    # 激活函数 ReLU, ELU, LeakyReLU, RReLU
                global_best["SKD"][4],      # 池化层 max mean
                global_best["SKD"][5:7],    # 优化器 Adamax, RMSprop, Adam, AdamW
                global_best["SKD"][7:9],    # 批次大小 4, 8, 16, 32
                global_best["lr_SKD"],      # 学习率 [0.00001, 0.001]
                global_best["SKD"][9],      # 批量归一化
                global_best["SKD"][10:12]   # dropout
            ]

            with open(output_file, "a") as file:
                file.write(f"第{it + 1}代：\n")
                file.write(f"参数：{parameters}\n")
                file.write(f"适应度值：{global_best['SKDfit']}\n\n")

        logging.info(f"第{it}代结果：{global_best['SKDfit']}")

    # 记录总运行时间
    total_time = time.time() - start_time
    with open(output_file, "a") as file:
        file.write(f"总运行时间：{total_time:.2f}秒\n")

    return parameters


if __name__ == "__main__":
    best_params = HLOCE_v3_0()
    logging.info("Best hyperparameters found:")
    logging.info(best_params)
