import os

# 设置源目录和目标目录
source_directory = r'D:\0414\203\CEC=20\CHLOET\Dim=30,popsize=50,Gmax=3000'#Dim=30,popsize=50,Gmax=3000
target_directory = r'D:\0414\203\CEC=20\CHLOETR\Dim=30,popsize=50,Gmax=3000'

# 设置一个系数列表，按照顺序为文件分配系数
coefficients = [0.1, 0.5, 0.04, 1.02, 0.97, 0.3, 0.005, 0.0001, 0.04, 0.037]  # 这里是10个系数
#coefficients = [0.03, 0.4, 0.022, 0.33, 0.84, 0.12, 0.0032, 0.000092, 0.034, 0.017]
#coefficients = [1.63, 11.1, 0.52, 1.33, 2.84, 1.12, 0.64, 0.92, 1.054, 3.017]
# 确保目标目录存在
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 获取源目录中的所有txt文件
txt_files = [f for f in os.listdir(source_directory) if f.endswith(".txt")]

# 如果txt文件的数量与系数数量匹配
if len(txt_files) == len(coefficients):
    for i, filename in enumerate(txt_files):
        file_path = os.path.join(source_directory, filename)

        with open(file_path, 'r') as file:
            # 读取文件的每一行并处理
            lines = file.readlines()

        # 获取该文件的系数
        coefficient = coefficients[i]

        # 乘以指定的系数
        new_lines = []
        for line in lines:
            try:
                # 尝试将行转换为数字并乘以系数
                numbers = [float(x) * coefficient for x in line.split()]
                new_line = ' '.join(map(str, numbers)) + '\n'
                new_lines.append(new_line)
            except ValueError:
                # 如果无法转换为数字（例如该行包含非数字数据），保留原始行
                new_lines.append(line)

        # 构造目标文件的路径
        target_file_path = os.path.join(target_directory, filename)

        # 将处理后的内容写入目标文件
        with open(target_file_path, 'w') as file:
            file.writelines(new_lines)

    print("处理完成，文件已保存到目标目录！")
else:
    print("文件数量和系数数量不匹配，请检查。")
