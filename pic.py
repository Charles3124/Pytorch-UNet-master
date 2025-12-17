import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['U-Net', 'Unet++', 'Attention Unet', 'SERUHC']
metrics = ['Dice', 'IoU', 'Accuracy', 'Sensitivity']

# 指标数据（按顺序：Dice, IoU, Accuracy, Sensitivity）
data = [
    [85.4, 75.4, 91.7, 92.5],
    [89.2, 80.5, 94.1, 96.9],
    [90.8, 83.1, 95.9, 98.3],
    [91.2, 83.8, 96.2, 97.7]
]

data = np.array(data)
bar_width = 0.18
x = np.arange(len(models))

# 颜色映射
colors = ['royalblue', 'mediumseagreen', 'gold', 'tomato']

plt.figure(figsize=(10, 6))
max_bar_height = 0  # 记录最高柱状图高度，用于自动设置 Y 轴上限
for i in range(len(metrics)):
    bars = plt.bar(x + i * bar_width, data[:, i], width=bar_width, label=metrics[i], color=colors[i])
    max_val = np.max(data[:, i])
    for j, bar in enumerate(bars):
        height = bar.get_height()
        max_bar_height = max(max_bar_height, height)
        # 添加数值标签
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        # 添加星标标记最高值
        # if np.isclose(height, max_val, atol=0.01):
        #     plt.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
        #              '★', ha='center', va='bottom', fontsize=14, color='black')

# 设置坐标轴与标签
plt.xticks(x + bar_width * 1.5, models)
plt.ylim(60, max_bar_height + 4)  # 自动留白，防止星标被截断
plt.ylabel("Performance (%)")

# 图例横向显示并靠右放置
plt.legend(loc='lower right', bbox_to_anchor=(1, -0.15), ncol=4, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()