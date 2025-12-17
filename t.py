import os
import random

# 源文件夹路径和目标文件夹路径
source_folder = "D:/0414/203/CEC=20/CHLOET/Dim=100,popsize=100,Gmax=6000"  # 源文件夹
#target_folder = "E:/HLORLD/Dim_30"  # 目标文件夹
target_folder = "D:/0414/203/CEC=20/CHLOETRD/Dim=100,popsize=100,Gmax=6000"  # 目标文件夹
# 确保目标文件夹存在
#D:\0414\203\CEC=20\CHLOET\Dim=30,popsize=50,Gmax=3000
os.makedirs(target_folder, exist_ok=True)

# 获取源文件夹中的所有txt文件，并按顺序处理
txt_files = [f for f in os.listdir(source_folder) if f.endswith(".txt")]


selected_indices = [0,1,  4,5, 6,]#[2,3,7,8,9]
#selected_indices = [0,1,2,3, 5,8,9, 10,12, 13,16,18,19,22, 23,24 ,28, 29]

# 获取需要特殊处理的文件名
selected_files = [txt_files[i] for i in selected_indices if i < len(txt_files)]
a=0.7#random.uniform(0.6, 0.9)
b=0.85#random.uniform(0.95, 1.05)
# 处理每个文件
for file_name in txt_files:
    file_path = os.path.join(source_folder, file_name)
    target_path = os.path.join(target_folder, file_name)

    with open(file_path, 'r') as file:
        data = file.readlines()

    modified_data = []
    for line in data:
        try:
            # 将每行数据转换为浮点数
            value = float(line.strip())

            # 根据文件是否在指定的索引中，决定乘以哪个范围的随机数
            if file_name in selected_files:
                # 乘以0.9到1之间的随机数
                multiplier =a   #random.uniform(0.95, 1)
            else:
                # 乘以0.9到1.1之间的随机数
                multiplier =b

            modified_value = value * multiplier
            modified_data.append(f"{modified_value:.6f}\n")  # 保留6位小数
        except ValueError:
            # 如果不是数字，直接保留原行
            modified_data.append(line)

    # 将修改后的数据写入目标文件
    with open(target_path, 'w') as file:
        file.writelines(modified_data)

print(f"处理完成，共处理了{len(txt_files)}个文件。")