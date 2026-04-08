"""
HLOE_test_v1_0.py

功能: 使用有效信息 HLO (HLOE) 对超参数调优
时间: 2026/04/08
版本: 1.0
备注:
1. HLOEOptimizer 用于封装 HLOE 的参数和迭代过程
2. 函数 testFunction 为外部函数，需要自行实现
   testFunction 需要接收 List[np.ndarray] 类型的 ndarray 列表
   并返回 List[float] 类型的数据，表示每个个体的适应度值，适应度越小表示越好
3. 根据解决问题的不同，bit 和 parameters（149行）需要根据实际情况修改
"""

import os
import time
import logging
from typing import Optional, Any, Union

import numpy as np

from traintest import testFunction


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class HLOEOptimizer:
    """HLOE 参数和迭代类"""

    def __init__(self, pop_size: int, bit: int):
        self.pop_size = pop_size
        self.bit = bit

        # 基本参数
        self.pr = np.zeros(pop_size)  # 随机学习概率
        self.pi = np.zeros(pop_size)  # 个体学习概率

        self.Kr = 2                   # 计算 pr[i]
        self.K, self.Ki = 0.88, 5     # 计算 pi[i]

        self.pr0, self.pi0 = 5 / bit, 2 / bit + 0.85   # total 为 0 时使用的 pr 和 pi 数值

    def update_population(
            self, popus: np.ndarray, IKD: np.ndarray,
            SKD: np.ndarray, IKDfits: np.ndarray
    ) -> np.ndarray:
        """HLOE 迭代"""
        # 更新种群
        for i in range(self.pop_size):
            # 计算当前 i 对应的 pr[i], pi[i], ps[i]
            total = np.sum(np.abs(IKD[i] - SKD))

            if total == 0:     # total 为 0 时，使用基本 HLO 的 pr 和 pi
                self.pr[i] = self.pr0
                self.pi[i] = self.pi0
            else:              # total 不为 0 时，使用有效信息的计算方法
                self.pr[i] = self.Kr / total
                self.pi[i] = self.K + self.Ki / total

            for j in range(self.bit):
                prob = np.random.uniform()
                if prob < self.pr[i]:      # 随机学习
                    popus[i][j] = np.random.randint(0, 2)
                elif prob < self.pi[i]:    # 个体学习
                    popus[i][j] = IKD[i][j]
                else:                      # 社会学习
                    popus[i][j] = SKD[j]

        return popus


def HLOE_v1_0(
        max_iter: int = 10,
        pop_size: int = 10,
        bit: int = 22,
        rl: int = 50
) -> Optional[list[Union[np.ndarray, np.int64]]]:
    """HLOE 优化超参数"""
    # 记录程序开始时间
    start_time = time.time()

    # 创建 HLOE 优化器，用于二进制超参数
    HLOE_optimiter = HLOEOptimizer(pop_size, bit)

    # 创建文件，保存最优解及运行时间
    output_file = f"HLOE_test_v1_0_results.txt"
    with open(output_file, "w") as file:
        file.write("HLOE 优化过程结果：\n")

    # 初始化种群
    pop: dict[str, Any] = {
        "popus": np.random.randint(0, 2, (pop_size, bit)),
        "fitness": None
    }

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
    count = np.zeros(pop_size)
    parameters = None

    # HLOE 迭代
    for it in range(max_iter):
        # HLOE 更新种群
        pop["popus"] = HLOE_optimiter.update_population(
            pop["popus"], individual["IKD"], global_best["SKD"], individual["IKDfits"]
        )

        # 更新适应度值
        pop["fitness"] = testFunction(pop["popus"])

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
                individual["IKDfits"][i] = testFunction([individual["IKD"][i]])[0]
                count[i] = 0

        # 寻找全局最优
        ind = np.argmin(individual["IKDfits"])
        if individual["IKDfits"][ind] < global_best["SKDfit"]:
            global_best["SKDfit"] = individual["IKDfits"][ind]
            global_best["SKD"] = individual["IKD"][ind].copy()

            # 记录当前最优解（需要根据实际情况修改）
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
    best_params = HLOE_v1_0()
    logging.info("Best hyperparameters found:")
    logging.info(best_params)
