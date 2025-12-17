import numpy as np
import cv2
import os

# 指定 npz 文件夹路径
npz_folder = 'D:/UNet_py/dataset_split/npzgood/'

# 指定保存目录
original_save_dir = 'D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/data2/original_images/'
label_save_dir = 'D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/data2/label_images/'

# 创建保存目录（如果不存在）
if not os.path.exists(original_save_dir):
    os.makedirs(original_save_dir)
if not os.path.exists(label_save_dir):
    os.makedirs(label_save_dir)

# 列出 npz 文件夹中的所有 npz 文件
npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]

# 遍历所有 npz 文件
for i, npz_filename in enumerate(npz_files):
    npz_filepath = os.path.join(npz_folder, npz_filename)

    # 加载 npz 文件
    npz_file = np.load(npz_filepath)

    # 提取图片数据
    original_image = npz_file['image']
    label_image = npz_file['label']

    # 检查数据形状和类型
    print(f'{npz_filename} - Original image shape: {original_image.shape}')
    print(f'{npz_filename} - Label image shape: {label_image.shape}')

    # 保存原图片
    original_image_name = f'image_{i + 1:02d}.jpg'
    original_image_path = os.path.join(original_save_dir, original_image_name)
    cv2.imwrite(original_image_path, original_image)

    # 直接保存标签图片，不进行处理
    label_image_name = f'image_{i + 1:02d}_mask.png'
    label_image_path = os.path.join(label_save_dir, label_image_name)
    cv2.imwrite(label_image_path, label_image)

    print(f"Original image saved at: {original_image_path}")
    print(f"Label image saved at: {label_image_path}")

print("All images have been successfully extracted and saved.")