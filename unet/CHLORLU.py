import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import random

import logging
from torch.utils.data import DataLoader
from pathlib import Path
from utils.data_loading import BasicDataset, CarvanaDataset, Custom_dataset
from evaluate import evaluate

# =================== 配置路径 ===================
model_path = "D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/checkpoints/modeltest.pth"
base_dir = Path('D:/UNet_py/dataset_split/npzgood')
list_dir = Path('D:/UNet_py/dataset_split/dataset_split')
weight_key = "outc.conv.weight"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== 数据加载器 ===================
split = "val"
db_test = Custom_dataset(base_dir, list_dir, split=split)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)


def save_if_better(model, modified_vals, weight_key, selected_indices, dice_score, model_path):
    global best_dice_score

    if dice_score > best_dice_score:
        best_dice_score = dice_score
        print(f"新最佳 Dice: {dice_score:.4f}，正在保存模型...")

        # 更新权重
        state_dict = model.state_dict()
        flat_weights = state_dict[weight_key].flatten().detach().cpu().numpy()
        for idx, val in zip(selected_indices, modified_vals):
            flat_weights[idx] = val
        updated_tensor = torch.tensor(flat_weights.reshape(state_dict[weight_key].shape), dtype=torch.float32)
        state_dict[weight_key] = updated_tensor
        model.load_state_dict(state_dict)

        # 保存新模型
        save_name = f"model_optimized_best_dice_{dice_score:.4f}.pth"
        save_path = os.path.join(os.path.dirname(model_path), save_name)
        torch.save(model, save_path)
        print(f"已保存至 {save_path}")
# =================== 全局变量保存索引 ===================
selected_indices = []
best_dice_score = -np.inf  # 全局最优 Dice（越高越好）
# =================== 连续优化函数 ===================
# def continuous_optimizer(obj_func, dim, bounds, max_gen=50, pop_size=10, rl=50,
#                          pr=0.005, pi=0.80, K1=0.15, K2=0.3, K3=0.75):
#     xmin, xmax = bounds
#     pop = xmin + np.random.rand(pop_size, dim) * (xmax - xmin)
#     IKD = pop.copy()
#     IKDfits = np.apply_along_axis(obj_func, 1, IKD)
#     SKD = IKD[np.argmin(IKDfits)].copy()
#     SKDfit = IKDfits.min()
#     count = np.zeros(pop_size)
#
#     for gen in range(max_gen):
#         for i in range(pop_size):
#             for j in range(dim):
#                 prob = random.random()
#                 if prob <= pr:
#                     pop[i][j] = xmin + random.random() * (xmax - xmin)
#                 elif prob < pi:
#                     pop[i][j] = np.random.normal(IKD[i][j], K1 * (SKD[j] - IKD[i][j]))
#                 else:
#                     perturb = K3 * random.random() * (SKD[j] - IKD[i][j])
#                     direction = 1 if random.random() < 0.5 else -1
#                     pop[i][j] = np.random.normal(SKD[j], K2 * (SKD[j] - IKD[i][j])) + direction * perturb
#
#         pop = np.clip(pop, xmin, xmax)
#         fits = np.apply_along_axis(obj_func, 1, pop)
#
#         for i in range(pop_size):
#             if fits[i] < IKDfits[i]:
#                 IKDfits[i] = fits[i]
#                 IKD[i] = pop[i].copy()
#                 count[i] = 0
#             else:
#                 count[i] += 1
#
#             if count[i] == rl:
#                 IKD[i] = xmin + np.random.rand(dim) * (xmax - xmin)
#                 IKDfits[i] = obj_func(IKD[i])
#                 count[i] = 0
#
#         best_idx = np.argmin(IKDfits)
#         if IKDfits[best_idx] < SKDfit:
#             SKDfit = IKDfits[best_idx]
#             SKD = IKD[best_idx].copy()
#
#     return SKD, SKDfit  #最优权值与适应度


def continuous_optimizer(obj_func, dim, bounds, max_gen=50, pop_size=10, rl=50,
                         pr=0.005, pi=0.80, ps=0.94,
                         Ki=0.15, Kmin=0.2, Kmax=0.8,
                         Ks1=0.5, Ks2=0.3, seta=0.6):
    xmin, xmax = bounds
    pop = xmin + np.random.rand(pop_size, dim) * (xmax - xmin)
    IKD = pop.copy()
    IKDfits = np.apply_along_axis(obj_func, 1, IKD)
    SKD = IKD[np.argmin(IKDfits)].copy()
    SKDfit = IKDfits.min()
    count = np.zeros(pop_size)

    def calculate_ranks(data, reverse=True):
        return np.argsort(np.argsort(-data if reverse else data)) + 1

    for gen in range(max_gen):
        # Spearman rank correlation
        D = np.sum((IKD - SKD) ** 2, axis=1)
        rankdis = calculate_ranks(D, reverse=True)
        rankfit = calculate_ranks(IKDfits, reverse=True)
        diff = rankdis - rankfit
        p_spearman = 1 - (6 * np.sum(diff ** 2)) / (pop_size * (pop_size ** 2 - 1))

        # Population update
        for i in range(pop_size):
            r1 = np.random.choice([x for x in range(pop_size) if x != i])
            # dist1 = np.sum((SKD - IKD[i]) ** 2)
            # dist2 = np.sum((SKD - IKD[r1]) ** 2)

            for j in range(dim):
                prob = np.random.rand()
                if prob <= pr:
                    pop[i][j] = xmin + np.random.rand() * (xmax - xmin)
                elif prob < pi:
                    sigma = abs(Ki * (SKD[j] - IKD[i][j]))
                    pop[i][j] = np.random.normal(IKD[i][j], sigma)
                elif prob < ps:
                    d1 = abs(SKD[j] - IKD[i][j])
                    d2 = abs(SKD[j] - IKD[r1][j])
                    d3 = abs((IKD[i][j] + SKD[j] + IKD[r1][j]) / 3.0 - IKD[i][j])

                    if p_spearman > seta:
                        sigma = Kmin * min(d1, d2)
                        pop[i][j] = np.random.normal(SKD[j], sigma)
                    else:
                        sigma = Kmax * d3
                        center = (IKD[r1][j] + SKD[j] + IKD[i][j]) / 3.0
                        pop[i][j] = np.random.normal(center, sigma)
                else:
                    direction = 1 if np.random.rand() < 0.5 else -1
                    perturb = direction * Ks1 * np.random.rand() * (SKD[j] - IKD[i][j])
                    sigma = Ks2 * (SKD[j] - IKD[i][j])
                    pop[i][j] = perturb + np.random.normal(SKD[j], sigma)

        # Clipping boundaries
        pop = np.clip(pop, xmin, xmax)

        # Evaluate fitness
        fits = np.apply_along_axis(obj_func, 1, pop)

        # Update individuals
        for i in range(pop_size):
            if fits[i] < IKDfits[i]:
                IKD[i] = pop[i].copy()
                IKDfits[i] = fits[i]
                count[i] = 0
            else:
                count[i] += 1
                if count[i] >= rl:
                    IKD[i] = xmin + np.random.rand(dim) * (xmax - xmin)
                    IKDfits[i] = obj_func(IKD[i])
                    count[i] = 0

        # Update global best
        best_idx = np.argmin(IKDfits)
        if IKDfits[best_idx] < SKDfit:
            SKDfit = IKDfits[best_idx]
            SKD = IKD[best_idx].copy()

    return SKD, SKDfit

# =================== 模型适应度函数 ===================
def model_fitness_from_weights(modified_values, model_path, weight_key, test_loader, device, metric="dice"):
    global selected_indices
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        state_dict = model.state_dict()
        original_weights = state_dict[weight_key].clone().cpu().numpy().flatten()

        # 替换指定索引
        for idx, val in zip(selected_indices, modified_values):
            original_weights[idx] = val

        # 更新权重
        updated_tensor = torch.tensor(original_weights.reshape(state_dict[weight_key].shape), dtype=torch.float32)
        state_dict[weight_key] = updated_tensor
        model.load_state_dict(state_dict)

        # 执行评估（假设 evaluate 返回 dict 含 dice）
        result = evaluate(model, test_loader, device=device,split="val")
        score = result["dice"] if isinstance(result, dict) else result
        save_if_better(model, modified_values, weight_key, selected_indices, score, model_path)
        return 1.0 - score  # HLO仍保持最小化优化机制

# =================== 主函数：优化模型输出权重 ===================
def optimize_model(model_path, weight_key, test_loader, device, split,  num_params=8, bounds=(-0.795, 0.810)):
    global selected_indices

    # 获取模型权重并展开
    model = torch.load(model_path, map_location='cpu')  # 加载模型对象
    model = model.to(device)
    # split = "val"
    db_test = Custom_dataset(base_dir, list_dir, split=split)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    evaluate(model, testloader, device, split)
    state_dict = model.state_dict()  # 提取其权重字典
    flat_weights = state_dict[weight_key].flatten().detach().cpu().numpy()

    total_len = len(flat_weights)
    # print(f"num_params={num_params}, type={type(num_params)}")

    # 随机选取索引
    selected_indices = sorted(random.sample(range(total_len), num_params))
    original_vals = flat_weights[selected_indices].copy()

    # 优化权重
    best_vals, best_fit = continuous_optimizer(
        obj_func=lambda x: model_fitness_from_weights(x, model_path, weight_key, test_loader, device),
        dim=num_params,
        bounds=bounds
    )

    print("被优化权重索引:", selected_indices)
    print("原始权重:", original_vals)
    print("优化结果:", best_vals)
    print("最优适应度（1 - Dice）:", best_fit)
    print("最佳Dice得分:", 1 - best_fit)

    # 可选：保存更新模型
    # state_dict[weight_key] = torch.tensor(flat_weights.reshape(state_dict[weight_key].shape), dtype=torch.float32)
    # torch.save(state_dict, "model_optimized.pth")

# =================== 执行入口 ===================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimize_model(model_path, weight_key, testloader,device, "val",  8, (-0.795, 0.810))
