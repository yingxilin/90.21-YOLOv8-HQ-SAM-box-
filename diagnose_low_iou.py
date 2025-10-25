#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose Low-IoU Cases (健壮版)
兼容 FungiTastic parquet ['label','file_name','width','height','rle']
可自动跳过损坏或格式不规范的 RLE。
"""

import os
import ast
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ========================== 配置路径 ==========================
GT_PARQUET = r"D:\search\fungi\26\data\masks\FungiTastic-Mini-ValidationMasks.parquet"
PRED_MASKS_DIR = r"D:\search\fungi\26\FungiTastic\out\masks_yolo_hqsam"
OUT_REPORT = r"D:\search\fungi\26\FungiTastic\out\diagnose_low_iou.csv"
IOU_THRESHOLD = 0.5


# ========================== 工具函数 ==========================
def safe_parse_rle(rle_data):
    """安全解析 RLE，无论是字符串、列表还是 ndarray"""
    if isinstance(rle_data, str):
        try:
            rle = ast.literal_eval(rle_data)
        except Exception:
            # 有些是逗号分隔字符串
            parts = [int(x) for x in rle_data.replace("[", "").replace("]", "").split(",") if x.strip().isdigit()]
            rle = parts
    elif isinstance(rle_data, (list, np.ndarray)):
        rle = list(map(int, rle_data))
    else:
        raise ValueError(f"Unsupported RLE type: {type(rle_data)}")
    return np.array(rle, dtype=int)


def decode_rle(rle, height, width):
    """解码 RLE 掩码"""
    mask = np.zeros(height * width, dtype=np.uint8)
    if len(rle) % 2 != 0:
        rle = rle[: len(rle) - 1]  # 修正奇数长度
    try:
        pairs = rle.reshape(-1, 2)
    except Exception:
        raise ValueError(f"Invalid RLE length: {len(rle)}")
    for start, length in pairs:
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')


def load_gt_dict(parquet_path):
    """加载 ground truth 掩码字典"""
    df = pd.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df)} rows from {os.path.basename(parquet_path)}")

    gt_dict = {}
    grouped = df.groupby("file_name")

    for fname, group in tqdm(grouped, desc="Decoding GT"):
        height = int(group.iloc[0]["height"])
        width = int(group.iloc[0]["width"])
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for _, row in group.iterrows():
            try:
                rle = safe_parse_rle(row["rle"])
                mask = decode_rle(rle, height, width)
                combined_mask = np.maximum(combined_mask, mask)
            except Exception as e:
                print(f"⚠️ Skip {fname} ({row['label']}): {e}")
                continue

        base = Path(fname).stem
        gt_dict[base] = combined_mask

    print(f"✅ Loaded {len(gt_dict)} GT masks.")
    return gt_dict


def calc_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


# ========================== 主逻辑 ==========================
def main():
    gt_dict = load_gt_dict(GT_PARQUET)
    results = []

    pred_files = list(Path(PRED_MASKS_DIR).glob("*.png"))
    print(f"Found {len(pred_files)} predicted masks.")

    for pred_path in tqdm(pred_files, desc="Evaluating IoU"):
        name = pred_path.stem
        if name not in gt_dict:
            continue

        gt_mask = gt_dict[name]
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            continue
        pred_mask = (pred_mask > 127).astype(np.uint8)

        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        iou = calc_iou(pred_mask, gt_mask)
        results.append((name, iou))

    df = pd.DataFrame(results, columns=["image_name", "iou"])
    mean_iou = df["iou"].mean() if not df.empty else 0
    print(f"\n=== Evaluation Report ===")
    print(f"Mean IoU: {mean_iou:.6f}")
    print(f"Total evaluated: {len(df)}")

    low_iou = df[df["iou"] < IOU_THRESHOLD]
    print(f"Low IoU (<{IOU_THRESHOLD}): {len(low_iou)} cases")

    out_path = Path(OUT_REPORT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    low_iou.to_csv(out_path, index=False)
    print(f"Low-IoU report saved to {out_path}")


if __name__ == "__main__":
    main()
