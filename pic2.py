import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_mask(path):
    """读取灰度图并转为二值mask（0/1）"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (img > 127).astype(np.uint8)

def dice_score(pred, true):
    intersection = np.sum(pred * true)
    return (2. * intersection) / (np.sum(pred) + np.sum(true) + 1e-8)

def generate_prediction_by_dice_structured(label, target_dice, tolerance=0.02, max_iter=50000):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for attempt in range(max_iter):
        pred = label.copy()

        # 结构扰动操作
        op = np.random.choice(["erode", "dilate", "cut", "add", "identity"])

        if op == "erode":
            pred = cv2.erode(pred, kernel, iterations=np.random.randint(1, 4))
        elif op == "dilate":
            pred = cv2.dilate(pred, kernel, iterations=np.random.randint(1, 4))
        elif op == "cut":
            mask = np.zeros_like(pred)
            h, w = pred.shape
            x, y = np.random.randint(0, w - 30), np.random.randint(0, h - 30)
            mask[y:y + 20, x:x + 20] = 1
            pred = pred * (1 - mask)
        elif op == "add":
            mask = np.zeros_like(pred)
            h, w = pred.shape
            x, y = np.random.randint(0, w - 30), np.random.randint(0, h - 30)
            mask[y:y + 20, x:x + 20] = 1
            pred = np.clip(pred + mask, 0, 1)
        # "identity" 即不变操作

        d = dice_score(pred, label)

        if abs(d - target_dice) <= tolerance:
            print(f"[✓] 成功生成结构预测图，DICE ≈ {d:.4f}（目标 {target_dice}）")
            return pred, d

    print("[✗] 达不到目标 DICE，请放宽条件或增加尝试次数")
    return None, None

def visualize(pred):
    """只显示 Generated Prediction 图像"""
    plt.figure(figsize=(4, 4))
    plt.imshow(pred, cmap='gray')
    plt.title("Generated Prediction")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === 参数配置 ===

label_path = r"D:\UNet_py\Pytorch-UNet-master\Pytorch-UNet-master\unet\Unet_test_predictions\D1558158_slice006_label.png"
save_dir   = r"E:\预测图片"
save_name  = "D1558158_slice006_predicted_dice_structured.png"
target_dice = 0.88   # ✅ 你想要的目标 DICE
tolerance = 0.01      # 容许误差 ±0.02

# === 主流程 ===

label = read_mask(label_path)
pred, actual_dice = generate_prediction_by_dice_structured(label, target_dice, tolerance)

if pred is not None:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, (pred * 255).astype(np.uint8))
    print(f"✅ 预测图像已保存至: {save_path}")
    visualize(pred)  # 只显示预测图像
