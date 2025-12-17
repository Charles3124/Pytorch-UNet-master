import os
import re
import shutil

def organize_images(folder_path):
    # 定义目标文件夹
    target_folders = {
        "label": os.path.join(folder_path, "label"),
        "original": os.path.join(folder_path, "original"),
        "overlay": os.path.join(folder_path, "overlay"),
        "prediction": os.path.join(folder_path, "prediction")
    }

    # 创建目标文件夹
    for folder in target_folders.values():
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 定义正则表达式来匹配文件名
    pattern = re.compile(r'_label\.png$|_original\.png$|_overlay\.png$|_prediction\.png$')

    # 遍历文件
    for file in files:
        match = pattern.search(file)
        if match:
            # 获取文件名中的标识
            if "_label.png" in file:
                target_folder = target_folders["label"]
            elif "_original.png" in file:
                target_folder = target_folders["original"]
            elif "_overlay.png" in file:
                target_folder = target_folders["overlay"]
            elif "_prediction.png" in file:
                target_folder = target_folders["prediction"]
            else:
                continue

            # 移动文件到目标文件夹
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(target_folder, file)
            shutil.move(src_path, dst_path)

    print("文件已成功分类到相应的文件夹中。")

# 使用示例
folder_path = r"D:\UNet_py\Pytorch-UNet-master\Pytorch-UNet-master\unet\test_predictions"
organize_images(folder_path)