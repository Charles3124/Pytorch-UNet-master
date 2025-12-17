
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import torch.serialization

# 定义模型路径（使用原始字符串处理特殊字符）
model_path = r"D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/checkpoints/modeltest.pth"

# ---- 新增：检查文件是否存在 ----
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在：{model_path}")

model = torch.load(model_path, weights_only=False)
print("模型参数键列表（直接加载对象）：")
for key in model.state_dict().keys():
    print(f" - {key}")
# 显示输出层权重和偏置
state_dict = model.state_dict()
weight_tensor = state_dict["outc.conv.weight"]
bias_tensor = model.state_dict()["outc.conv.bias"]
print("偏置形状:", bias_tensor.shape)
print("\n输出层偏置（值）:", state_dict["outc.conv.bias"])
# 查看张量形状
print("权重形状:", weight_tensor.shape)  # 输出应为 torch.Size([2, 32, 1, 1])

# 将权重转换为NumPy数组（可选，便于索引操作）
weight_np = weight_tensor.detach().cpu().numpy()

# 按输出通道和输入通道遍历权重
for out_channel in range(weight_tensor.size(0)):  # 遍历输出通道（0~1）
    print(f"\n输出通道 {out_channel} 的权重：")
    for in_channel in range(weight_tensor.size(1)):  # 遍历输入通道（0~31）
        # 提取权重值（由于卷积核是1x1，直接取标量值）
        value = weight_tensor[out_channel, in_channel, 0, 0].item()
        print(f"输入通道 {in_channel:2d}: {value:.6f}")
# print("输出层权重（形状）:", state_dict["outc.conv.weight"].shape)
# print("输出层权重（部分值）:\n", state_dict["outc.conv.weight"][0, 0, :3, :3])  # 示例：显示前3x3的值
# print("\n输出层偏置（值）:", state_dict["outc.conv.bias"])
