from PIL import Image
import numpy as np

def change_white_border_to_black(image_path, output_path, tolerance=10):
    # 打开图片
    img = Image.open(image_path).convert('RGBA')
    data = np.array(img)

    # 白色定义（带有容差）
    white_criteria = (data[..., :3] >= 255 - tolerance).all(axis=-1)

    # 找到边缘的白色区域并改为黑色
    rows, cols = data.shape[:2]

    # 上边缘
    for row in range(rows):
        if not white_criteria[row].all():
            break
        data[row][white_criteria[row]] = [0, 0, 0, 255]

    # 下边缘
    for row in range(rows - 1, -1, -1):
        if not white_criteria[row].all():
            break
        data[row][white_criteria[row]] = [0, 0, 0, 255]

    # 左边缘
    for col in range(cols):
        if not white_criteria[:, col].all():
            break
        data[:, col][white_criteria[:, col]] = [0, 0, 0, 255]

    # 右边缘
    for col in range(cols - 1, -1, -1):
        if not white_criteria[:, col].all():
            break
        data[:, col][white_criteria[:, col]] = [0, 0, 0, 255]

    # 保存结果
    new_img = Image.fromarray(data)
    new_img.save(output_path)

# 使用示例
change_white_border_to_black(r'E:\预测图片\UNet++.png', r'E:\预测图片\UNet++_black.png')
