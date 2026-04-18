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
from typing import Optional, Any, Union

import numpy as np

from traintest import testFunction


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class HLOCEOptimizer:
    """HLOCE 参数和迭代类"""

    def __init__(self, pop_size: int, bit: int, ne_ratio: float = 0.18):
        self.pop_size = pop_size
        self.bit = bit
        self.ne = int(self.pop_size * ne_ratio)

        # 伯努利参数初始化
        self.ber_params_before = self._initialize_multinoulli_parameters(bit)

        # 基本参数
        self.a = 0.83    # 交叉熵参数

        self.pr = np.zeros(pop_size)  # 随机学习概率
        self.pi = np.zeros(pop_size)  # 个体学习概率
        self.ps = np.zeros(pop_size)  # 决定交叉熵学习还是社会学习

        self.Kr, self.prMax = 2, 0.1                   # 计算 pr[i]
        self.K1, self.Ki, self.piMax = 0.86, 4, 0.90   # 计算 pi[i]
        self.K2, self.Ks, self.psMax = 0.24, 6, 0.32   # 计算 ps[i]

        self.pr0, self.pi0, self.ps0 = 0.005, 0.80, 0.88  # sum 为 0 时的取值

    def update_population(
            self, popus: np.ndarray, IKD: np.ndarray,
            SKD: np.ndarray, IKD_fits: np.ndarray
    ) -> np.ndarray:
        """HLOCE 迭代"""
        # 计算交叉熵概率
        ber_params = self._ce_prob(IKD_fits, IKD, self.ne)

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
    def _ce_prob(IKD_fits: np.ndarray, IKD: np.ndarray, ne: int) -> np.ndarray:
        """使用多元伯努利分布计算交叉熵概率"""
        indices = np.argpartition(IKD_fits, ne)[:ne]       # 找到适应度最小的 ne 个个体索引
        elite_population = IKD[indices]                    # 提取精英个体
        probabilities = np.mean(elite_population, axis=0)  # 计算每个基因位点为 1 的概率
        return probabilities


class CHLOCEOptimizer:
    """CHLOCE 参数和迭代类"""

    def __init__(
            self, pop_size: int, dim: int,
            params_min: np.array, params_max: np.array,
            ne_ratio: float = 0.22
    ):
        self.pop_size = pop_size
        self.dim = dim
        self.ne = int(self.pop_size * ne_ratio)
        self.params_min = params_min
        self.params_max = params_max

        # 高斯参数初始化
        self.gaussian_params_before = self._initialize_gaussian_parameters(dim)

        # 基本参数
        self.a = 0.8    # 交叉熵参数

        self.K1, self.K2, self.K3 = 0.1, 0.8, 0.4

        self.pr = 0.005
        self.pi = np.zeros(self.pop_size)
        self.ps = np.zeros(self.pop_size)

        self.K1, self.Ki = 0.84, 0.20
        self.K2, self.Ks = 0.70, 0.19

        self.pi0, self.ps0 = 0.80, 0.71

    def update_population(
            self, popus: np.ndarray, IKD: np.ndarray,
            SKD: np.ndarray, IKD_fits: np.ndarray
    ) -> np.ndarray:
        """CHLOCE 迭代"""
        # 交叉熵高斯参数更新
        gaussian_params = self._ce_gaussian(IKD, IKD_fits, self.ne)
        gaussian_params_after = []

        for j in range(self.dim):
            current = gaussian_params[j]
            previous = self.gaussian_params_before[j]

            mean = self.a * current[0] + (1 - self.a) * previous[0]
            std = self.a * current[1] + (1 - self.a) * previous[1]
            gaussian_params_after.append([mean, std])

        self.gaussian_params_before = [gp.copy() for gp in gaussian_params_after]

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
                self.ps[i] = self.ps0
            else:
                self.pi[i] = self.K1 + self.Ki * sum_diff / self.dim
                self.ps[i] = self.K2 + self.Ks * sum_diff / self.dim

            # 更新种群每个维度
            for j in range(self.dim):
                prob = np.random.rand()
                if prob < self.pr:                     # 随机学习
                    popus[i][j] = self.params_min[j] + np.random.rand() * (self.params_max[j] - self.params_min[j])
                elif prob < self.pi[i]:                # 个体学习
                    popus[i][j] = np.random.normal(IKD[i][j], self.K1 * abs(SKD[j] - IKD[i][j]))
                else:
                    if np.random.rand() < self.ps[i]:  # 交叉熵学习
                        mean, std = gaussian_params_after[j]
                        popus[i][j] = np.random.normal(mean, std)
                    else:                              # 社会学习
                        direction = SKD[j] - IKD[i][j]
                        sign = 1 if np.random.rand() < 0.5 else -1
                        popus[i][j] = (
                                sign * self.K2 * np.random.rand() * direction
                                + np.random.normal(SKD[j], self.K3 * abs(direction))
                        )

                popus[i][j] = np.clip(popus[i][j], self.params_min[j], self.params_max[j])

        return popus

    @staticmethod
    def _initialize_gaussian_parameters(dim: int) -> list[list[float]]:
        """初始化高斯参数"""
        return [[np.random.rand(), np.random.rand()] for _ in range(dim)]

    @staticmethod
    def _ce_gaussian(IKD: np.ndarray, IKD_fits: np.ndarray, ne: int) -> list[list[float]]:
        """计算交叉熵高斯参数"""
        indices = np.argpartition(IKD_fits, ne)[:ne]
        elite = IKD[indices]
        mean = np.mean(elite, axis=0)
        std = np.std(elite, axis=0)
        return [[m, s] for m, s in zip(mean, std)]


def HLOCE_v3_0(
        max_iter: int = 20,
        pop_size: int = 20,
        bit: int = 16,
        dim: int = 2,
        rl: int = 50,
        use_attention: bool = True
) -> Optional[list[Union[np.ndarray, np.int64]]]:
    """HLOCE 优化二进制超参数，CHLOCE 优化学习率"""
    # 记录程序开始时间
    start_time = time.time()

    # 创建 HLOCE 优化器，用于二进制超参数
    HLOCE_optimiter = HLOCEOptimizer(pop_size, bit)

    # 创建 CHLOCE 优化器，用于学习率
    learning_rate_min = 1e-5
    learning_rate_max = 1e-3
    attention_ratio_min = 0.125
    attention_ratio_max = 1.0
    params_min = np.array([learning_rate_min, attention_ratio_min])
    params_max = np.array([learning_rate_max, attention_ratio_max])

    CHLOCE_optimiter = CHLOCEOptimizer(pop_size, dim, params_min, params_max)

    # 创建输出文件
    output_file = f"HLOCE_test_v3_0_results_{'attention' if use_attention else 'baseline'}.txt"
    with open(output_file, "w") as file:
        file.write("HLOCE + CHLOCE 优化过程结果：\n")

    # 种群初始化
    # HLOCE 负责二进制种群，CHLOCE 负责实数种群
    # testFunction 对二进制 + 实数种群共同评价，HLOCE 和 CHLOCE 共用适应度值 fitness
    # IKD 含义是个体的历史最优值，SKD 含义是种群的全局最优值

    # 初始化 HLOCE 和 CHLOCE 种群
    pop: dict[str, Any] = {
        "HLOCE_pop": np.random.randint(0, 2, (pop_size, bit)),
        "CHLOCE_pop": params_min + np.random.rand(pop_size, dim) * (params_max - params_min),
        "fitness": None,
    }
    
    # 调用 testFunction，初始化适应度
    pop["fitness"] = testFunction(
        HLOCE_pop=pop["HLOCE_pop"],
        CHLOCE_pop=pop["CHLOCE_pop"],
        use_attention=use_attention
    )

    # 最优个体索引
    ind = np.argmin(pop["fitness"])

    # 初始化个体最优
    individual: dict[str, Any] = {
        "HLOCE_IKD": pop["HLOCE_pop"].copy(),
        "CHLOCE_IKD": pop["CHLOCE_pop"].copy(),
        "IKD_fits": pop["fitness"].copy(),
    }

    # 初始化全局最优
    global_best: dict[str, Any] = {
        "HLOCE_SKD": pop["HLOCE_pop"][ind].copy(),
        "CHLOCE_SKD": pop["CHLOCE_pop"][ind].copy(),
        "SKD_fit": pop["fitness"][ind],
    }

    # 初始化计数器
    count = np.zeros(pop_size)

    # HLOCE + CHLOCE 迭代
    parameters = None
    for it in range(max_iter):
        # HLOCE 更新二进制参数
        pop["HLOCE_pop"] = HLOCE_optimiter.update_population(
            pop["HLOCE_pop"], individual["HLOCE_IKD"], global_best["HLOCE_SKD"], individual["IKD_fits"]
        )

        # CHLOCE 更新学习率
        pop["CHLOCE_pop"] = CHLOCE_optimiter.update_population(
            pop["CHLOCE_pop"], individual["CHLOCE_IKD"], global_best["CHLOCE_SKD"], individual["IKD_fits"]
        )

        # 调用 testFunction，更新适应度
        pop["fitness"] = testFunction(
            HLOCE_pop=pop["HLOCE_pop"],
            CHLOCE_pop=pop["CHLOCE_pop"],
            use_attention=use_attention
        )

        # 更新个体最优
        for i in range(pop_size):
            if pop["fitness"][i] < individual["IKD_fits"][i]:
                individual["HLOCE_IKD"][i] = pop["HLOCE_pop"][i].copy()
                individual["CHLOCE_IKD"][i] = pop["CHLOCE_pop"][i].copy()
                individual["IKD_fits"][i] = pop["fitness"][i]
                count[i] = 0
            else:
                count[i] += 1

            # 重新初始化
            if count[i] == rl:
                individual["HLOCE_IKD"][i] = np.random.randint(0, 2, bit)
                individual["CHLOCE_IKD"][i] = np.array([
                    np.random.uniform(learning_rate_min, learning_rate_max),
                    np.random.uniform(attention_ratio_min, attention_ratio_max)
                ])
                individual["IKD_fits"][i] = testFunction(
                    HLOCE_pop=[individual["HLOCE_IKD"][i]],
                    CHLOCE_pop=[individual["CHLOCE_IKD"][i]],
                    use_attention=use_attention
                )[0]
                count[i] = 0

        # 寻找全局最优
        ind = np.argmin(individual["IKD_fits"])
        if individual["IKD_fits"][ind] < global_best["SKD_fit"]:
            global_best["HLOCE_SKD"] = individual["HLOCE_IKD"][ind].copy()
            global_best["CHLOCE_SKD"] = individual["CHLOCE_IKD"][ind].copy()
            global_best["SKD_fit"] = individual["IKD_fits"][ind]

            parameters = [
                # 网络结构参数
                global_best["HLOCE_SKD"][0:2],    # 滤波器数量 4, 8, 16, 32

                # 网络模块类型
                global_best["HLOCE_SKD"][2:4],    # 激活函数类型 ReLU, ELU, LeakyReLU, RReLU
                global_best["HLOCE_SKD"][4],      # 批量归一化 使用，不使用
                global_best["HLOCE_SKD"][5],      # 池化层类型 MaxPool2d, AvgPool2d

                # 训练超参数
                global_best["CHLOCE_SKD"][0],     # 学习率 [0.00001, 0.001]
                global_best["HLOCE_SKD"][6:8],    # 批量大小 4, 8, 16, 32

                # 正则化参数
                global_best["HLOCE_SKD"][8:10],   # 随机丢弃 不使用，Dropout，GaussianDropout

                # 优化器参数
                global_best["HLOCE_SKD"][10:12],  # 优化器类型 RMSprop, Adam, AdamW, Adamax

                # Attention 模块参数
                global_best["CHLOCE_SKD"][1],     # 中间通道数 [0.125, 1.0]
                global_best["HLOCE_SKD"][12],     # 输出激活函数 Sigmoid, Hardsigmoid
                global_best["HLOCE_SKD"][13],     # 融合方式 加法，拼接
                global_best["HLOCE_SKD"][14:16],  # Attention 启用深度
            ]

            with open(output_file, "a") as file:
                file.write(f"第{it + 1}代：\n")
                file.write(f"参数：{parameters}\n")
                file.write(f"适应度值：{global_best['SKD_fit']}\n\n")

        logging.info(f"第{it + 1}代结果：{global_best['SKD_fit']}")

    # 记录总运行时间
    total_time = time.time() - start_time
    with open(output_file, "a") as file:
        file.write(f"总运行时间：{total_time:.2f}秒\n")

    return parameters


if __name__ == "__main__":
    best_params = HLOCE_v3_0()
    logging.info("Best hyperparameters found:")
    logging.info(best_params)
