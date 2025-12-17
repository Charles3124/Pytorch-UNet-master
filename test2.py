
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import torch.serialization
import torch.nn as nn
# 定义模型路径（使用原始字符串处理特殊字符）
model_path = r"D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/checkpoints/modeltest.pth"

# ---- 新增：检查文件是否存在 ----
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在：{model_path}")

model = torch.load(model_path, weights_only=False)
print("模型参数键列表（直接加载对象）：")
for key in model.state_dict().keys():
    print(f" - {key}")

state_dict = model.state_dict()
new_weight = nn.init.xavier_normal_(torch.empty_like(state_dict["outc.conv.weight"]))
state_dict["outc.conv.weight"] = new_weight

# 示例2：直接赋值特定值（如全设为0.5）
# state_dict["outc.conv.weight"].data.fill_(0.5)

# 修改偏置（例如清零）
state_dict["outc.conv.bias"].data.zero_()

# 将修改后的参数加载回模型
model.load_state_dict(state_dict)

# ----------------------------
# 4. 保存修改后的模型
# ----------------------------
# 保存整个模型对象（含结构）
new_model_path = model_path.replace(".pth", "_modified.pth")
torch.save(model, new_model_path)
