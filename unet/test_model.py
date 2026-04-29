"""
test_model.py

功能: 测试模型
时间: 2026/04/03
版本: 1.0
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from torch.utils.data import DataLoader

from traintest import UNet, ROOT_DIR, PROJECT_DIR, BASE_DIR, LIST_DIR
from evaluate import evaluate
from utils.data_loading import Custom_dataset


def test_model(saved_models: str, test_save_dir: Optional[str] = None, split: str = "test_vol"):
    """测试模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型，提取超参数
    checkpoint = torch.load(saved_models, map_location=device, weights_only=False)
    hparams = {
        "n_channels": 3,
        "n_classes": 1,
        "use_attention": checkpoint["use_attention"],
        "bilinear": checkpoint["bilinear"],
        "blocks_number": checkpoint["blocks_number"],
        "filter_size": checkpoint["filter_size"],
        "filters_number": checkpoint["filters_number"],
        "activation": checkpoint["activation"],
        "use_batchnorm": checkpoint["use_batchnorm"],
        "pooling": checkpoint["pooling"],
        "learning_rate": checkpoint["learning_rate"],
        "batch_size": checkpoint["batch_size"],
        "use_dropout": checkpoint["use_dropout"],
        "optimizer_type": checkpoint["optimizer_type"],
        "attention_ratio": checkpoint["attention_ratio"],
        "attention_activation": checkpoint["attention_activation"],
        "attention_fusion": checkpoint["attention_fusion"],
    }

    if "attention_depth" in checkpoint:
        hparams["attention_depth"] = checkpoint["attention_depth"]
    else:
        hparams["attention_depth"] = 3

    model = UNet(hparams)

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    # 对训练完的模型进行测试
    db_test = Custom_dataset(BASE_DIR, LIST_DIR, split=split)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    test_results = evaluate(model, test_loader, device, split, test_save_path=test_save_dir)
    torch.cuda.empty_cache()

    return hparams, test_results


# 批量测试模型
if __name__ == "__main__":
    # 配置参数
    models_dir = f"{ROOT_DIR}/{PROJECT_DIR}/checkpoints/"
    excel_save_path = f"{ROOT_DIR}/{PROJECT_DIR}/model_test_results.xlsx"

    # 日志配置
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 获取所有模型文件
    model_dir_path = Path(models_dir)
    model_files = list(model_dir_path.glob("*.pth"))

    logging.info(f"找到 {len(model_files)} 个模型文件")

    # 存储结果
    results_list = []

    # 测试模型
    for i, model_path in enumerate(model_files, 1):
        # 不含扩展名的文件名
        model_name = model_path.stem
        logging.info(f"[{i}/{len(model_files)}] 测试模型: {model_name}")

        # 调用测试函数
        hparams, (
            dice, dice_std, iou, iou_std, acc, acc_std, rec, rec_std, pre, pre_std
        ) = test_model(saved_models=str(model_path))

        # 保存结果
        result = {
            "model_name": model_name,
            "dice": dice,
            "dice_std": dice_std,
            "mean_iou": iou,
            "std_iou": iou_std,
            "acc": acc,
            "acc_std": acc_std,
            "rec": rec,
            "rec_std": rec_std,
            "pre": pre,
            "pre_std": pre_std,
            "use_attention": hparams["use_attention"],
            "bilinear": hparams["bilinear"],
            "blocks_number": hparams["blocks_number"],
            "filter_size": hparams["filter_size"],
            "filters_number": hparams["filters_number"],
            "activation": hparams["activation"],
            "use_batchnorm": hparams["use_batchnorm"],
            "pooling": hparams["pooling"],
            "learning_rate": hparams["learning_rate"],
            "batch_size": hparams["batch_size"],
            "use_dropout": hparams["use_dropout"],
            "optimizer_type": hparams["optimizer_type"],
            "attention_ratio": hparams["attention_ratio"],
            "attention_activation": hparams["attention_activation"],
            "attention_fusion": hparams["attention_fusion"],
            "attention_depth": hparams["attention_depth"],
        }
        results_list.append(result)

    # 保存结果到 Excel
    df = pd.DataFrame(results_list)
    excel_save_path = Path(excel_save_path)
    excel_save_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_save_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Test Results", index=False)

        # 获取工作表并调整列宽
        worksheet = writer.sheets["Test Results"]
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    logging.info(f"结果已保存到: {excel_save_path}")


# 测试模型
# if __name__ == "__main__":
#     base_path = f"{ROOT_DIR}/{PROJECT_DIR}/checkpoints/"
#     saved_models = os.path.join(
#         base_path,
#         "model_dice_0.8857,params_[1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1]-attention.pth"
#     )
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     test_save_dir = f"{ROOT_DIR}/unet_test/"
#
#     if saved_models:
#         test_results = test_model(saved_models)
