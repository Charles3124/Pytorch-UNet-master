"""
BHLOtest.py

功能: 使用 BHLO 对 U-Net 超参数调优
时间: 2025/11/30
版本: 1.0
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from traintest import testFunction


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 输出4个动作
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)


def BHLO(MaxIter: int = 10, PopSize: int = 10, Bits: int = 24, rl: int = 50):
    """BHLO 调优 U-Net 超参数"""
    pr = np.full(PopSize, 4.7 / Bits)    # 各个体初始 pr
    pi = np.full(PopSize, 0.85)
    ps = np.full(PopSize, 0.94)

    # 初始化 DQN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 空种群
    pop = {"popus": None, "fitness": None, "Rpopus": None}
    individual = {"IKD": None, "IKDfits": None, "Individual_BtoR": None}
    globalBest = {"SKD": None, "SKDfit": np.inf}

    # 初始化种群
    pop["popus"] = np.random.randint(0, 2, (PopSize, Bits))  # 生成 [0, 2) 间的整数

    delta = [0.00035,- 0.00041, 0.00724, -0.00492, 0.00135,- 0.00147]

    # 创建文件，保存最优解及运行时间
    output_file = "BHLO_results.txt"
    with open(output_file, "w") as file:
        file.write("BHLO优化过程结果：\n")

    start_time = time.time()  # 记录程序开始时间

    pop["fitness"] = testFunction(pop["popus"])  # 得到不同参数模型损失
    prev_fitness = pop["fitness"].copy()

    individual["IKD"] = pop["popus"].copy()
    individual["IKDfits"] = pop["fitness"].copy()
    count = np.zeros(PopSize)    # 重新初始化计数器
    ind = np.argmin(pop["fitness"])
    globalBest["SKD"] = pop["popus"][ind].copy()
    globalBest["SKDfit"] = pop["fitness"][ind]

    def strategy_update_bit(i: int, j: int, IKD: np.ndarray, ind: int) -> int:
        # Step 1: 选取不重复个体索引 d[0] ~ d[3]
        d = [ind]
        while len(d) < 4:
            new_index = random.randint(0, PopSize - 1)
            if new_index not in d and new_index != i:
                d.append(new_index)

        # Step 2: 计算策略值 s
        s = IKD[d[1]][j] + IKD[d[2]][j] + IKD[d[3]][j]

        # Step 3: 映射 s 到参数 t
        t_mapping = {
            0: 0.005,
            1: 0.6633,
            2: 0.3367,
            3: 0.9995
        }
        t = t_mapping.get(s, 0.5)  # 默认中值概率

        # Step 4: 根据 t 概率生成 bit（0或1）
        return 1 if random.random() < t else 0

    # BHLO 迭代
    for it in range(0, MaxIter):
        # 更新种群
        for i in range(0, PopSize):
            for j in range(0, Bits):
                prob = np.random.uniform(0, 1, 1)
                if prob <= pr[i]:
                    pop["popus"][i][j] = np.random.randint(0, 2)
                elif prob < pi[i]:
                    pop["popus"][i][j] = individual["IKD"][i][j]
                elif prob < ps[i]:
                    pop["popus"][i][j] = globalBest["SKD"][j]
                else:
                    # 应用新策略函数
                    pop["popus"][i][j] = strategy_update_bit(i, j, individual["IKD"], ind)

        # 更新适应度值
        pop["fitness"] = testFunction(pop["popus"])

        dqn_actions = []
        for i in range(PopSize):
            previous = prev_fitness[i]
            current = pop["fitness"][i]
            improve = 1.0 if current < previous else 0.0
            state = torch.tensor([previous, current], dtype=torch.float32).to(device)
            q_values = net(state)
            action = torch.argmax(q_values).item()
            dqn_actions.append(action)

            if it % 55 == 0:   # 只有当代数是 55 的倍数时才更新参数
                if action == 0:
                    pi[i] += random.random() * delta[2]
                    ps[i] += random.random() * delta[4]
                elif action == 1:
                    pr[i] += random.random() * delta[0]
                    pi[i] += random.random() * delta[2]
                    ps[i] += random.random() * delta[4]
                elif action == 2:
                    pi[i] += random.random() * delta[3]
                    ps[i] += random.random() * delta[5]
                elif action == 3:
                    pr[i] += random.random() * delta[1]
                    pi[i] += random.random() * delta[3]
                    ps[i] += random.random() * delta[5]

            # DQN训练
            target = q_values.clone().detach()
            target[action] = improve
            prediction = net(state)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_fitness = pop["fitness"].copy()  # 更新前代适应度

        # 更新个体最优
        for i in range(0, PopSize):
            if pop["fitness"][i] < individual["IKDfits"][i]:
                individual["IKDfits"][i] = pop["fitness"][i]
                individual["IKD"][i] = pop["popus"][i].copy()
                count[i] = 0
            else:
                count[i] += 1

            # 重新初始化
            if count[i] == rl:
                individual["IKD"][i] = np.random.randint(0, 2, (1, Bits))
                individual["IKDfits"] = testFunction(individual["IKD"])
                count[i] = 0

        # 寻找全局最优
        ind = np.argmin(individual["IKDfits"])
        if individual["IKDfits"][ind] < globalBest["SKDfit"]:
            globalBest["SKDfit"] = individual["IKDfits"][ind]
            globalBest["SKD"] = individual["IKD"][ind].copy()

            # 记录当前最优解
            parameter = [
                globalBest["SKD"][0:2],    # 块数 3, 5, 7, 9
                globalBest["SKD"][2:4],    # 滤波器数量 4, 8, 16, 32
                globalBest["SKD"][4:6],    # 激活函数 ReLU, ELU, LeakyReLU, RReLU
                globalBest["SKD"][6],      # 池化层 max mean
                globalBest["SKD"][7:9],    # 优化器 Adamax, RMSprop, Adam, AdamW
                globalBest["SKD"][9:11],   # 批次大小 4, 8, 16, 32
                globalBest["SKD"][11:21],  # 学习率 [0.00001, 0.001]
                globalBest["SKD"][21],     # 批量归一化
                globalBest["SKD"][22:24]   # dropout
            ]

            # 将参数和适应度值写入文件
            with open(output_file, "a") as file:
                file.write(f"第{it + 1}代：\n")
                file.write(f"参数：{parameter}\n")
                file.write(f"适应度值：{globalBest['SKDfit']}\n\n")

        print(f"第{it}代结果：{globalBest['SKDfit']}")

    # 记录总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    with open(output_file, "a") as file:
        file.write(f"总运行时间：{total_time:.2f}秒\n")

    # 返回最优权值结果
    return parameter


if __name__ == "__main__":
    best_params = BHLO()
    print("Best hyperparameters found:")
    print(best_params)
