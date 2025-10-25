#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# evaluate_hqsam_iou_report.py
# 针对 YOLO+HQ-SAM 掩码进行真实 IoU 评估，并生成低 IoU 报告。

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import json
import sys
from typing import List, Tuple, Optional, Any

# ==============================================================================
# 核心 RLE 解码和 IoU 计算函数 (与您之前的脚本保持一致)
# ==============================================================================

def rle_decode_pairs_or_counts(arr: Any, H: int, W: int) -> Optional[np.ndarray]:
    """
    解码 RLE-like 数组（pairs 或 counts）。返回 uint8 mask (H, W)。
    """
    if arr is None:
        return None
    
    try:
        if isinstance(arr, (np.ndarray, list, tuple)):
            nums = [int(x) for x in arr]
        elif isinstance(arr, str):
            toks = [t for t in arr.replace(",", " ").split() if t.strip()]
            nums = [int(t) for t in toks]
        else:
            return None
    except Exception:
        return None

    if len(nums) == 0:
        return None

    total = H * W
    try:
        # 尝试 pairs-style (start, len, ...)
        if len(nums) % 2 == 0 and max(nums) > 0:
            s = sum(max(0, nums[i+1]) for i in range(0, len(nums), 2))
            if s > 0 and s < total * 10:
                first_start = nums[0]
                is_one_indexed = first_start >= 1
                mask_flat = np.zeros(total, dtype=np.uint8)
                for i in range(0, len(nums), 2):
                    start = int(nums[i]) - (1 if is_one_indexed else 0)
                    length = int(nums[i+1])
                    if length <= 0 or start < 0 or start >= total:
                        continue
                    end = min(start + length, total)
                    mask_flat[start:end] = 1
                if mask_flat.sum() > 0:
                    # 优先 F-order (Fortran) 约定，如果大小不匹配则尝试 C-order
                    return mask_flat.reshape((H, W), order='F') if mask_flat.size == total else mask_flat.reshape((H, W))

        # 尝试 counts-style [c1, c2, c3, ...]
        flat = []
        val = 0
        for c in nums:
            if c <= 0: continue
            flat.extend([val] * int(c))
            val = 1 - val
            
        flat = np.array(flat, dtype=np.uint8)
        
        if flat.size != total:
            out = np.zeros(total, dtype=np.uint8)
            out[:min(flat.size, total)] = flat[:min(flat.size, total)]
            flat = out

        # 默认使用 F-order
        chosen = flat.reshape((H, W), order='F')
        
        if chosen.sum() == 0:
            return None
        return chosen
    except Exception:
        return None

def normalize_name(p: str) -> str:
    return Path(p).stem.lower()

def build_gt_index(parquet_path: str) -> dict:
    """读取 parquet，构建映射: stem -> list of rows (rle, H, W)"""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise FileNotFoundError(f"无法读取 Parquet 文件 {parquet_path}: {e}")
        
    name_col, rle_col, h_col, w_col = None, None, None, None
    for c in df.columns:
        lc = c.lower()
        if lc in ("file_name", "filename", "image", "image_path", "image_id", "file"): name_col = c
        if lc in ("rle", "mask_rle", "segmentation", "encoded_pixels", "counts"): rle_col = c
        if lc in ("height", "img_h", "h", "image_height"): h_col = c
        if lc in ("width", "img_w", "w", "image_width"): w_col = c
        
    if name_col is None or rle_col is None:
        raise KeyError(f"找不到必要列 (文件名/RLE)。parquet 列: {list(df.columns)}")
        
    if h_col is None: df['__h'] = 0; h_col = '__h'
    if w_col is None: df['__w'] = 0; w_col = '__w'

    index = {}
    grouped = df.groupby(name_col)
    for fname, g in grouped:
        key = normalize_name(str(fname))
        rows = []
        for _, row in g.iterrows():
            rows.append((
                row[rle_col], 
                int(row[h_col]) if row[h_col] > 0 else 0, 
                int(row[w_col]) if row[w_col] > 0 else 0
            ))
        index[key] = rows
    return index

def compute_iou_np(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算两个 uint8 二值掩码的 IoU"""
    p = (np.asarray(pred_mask) > 127).astype(np.uint8)
    g = (np.asarray(gt_mask) > 127).astype(np.uint8)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# ==============================================================================
# 主逻辑
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate mask IoU and generate low IoU report.")
    parser.add_argument(
        "--pred_dir", 
        required=False,
        default=r"D:\search\fungi\26\FungiTastic\out\masks_yolo_hqsam",
        help="预测掩码 PNG 目录（默认使用 YOLO+HQ-SAM 路径）"
    )
    parser.add_argument(
        "--gt_parquet", 
        required=False,
        default=r"D:\search\fungi\26\data\masks\FungiTastic-Mini-ValidationMasks.parquet",
        help="GT parquet 路径 (默认使用 ValidationMasks)"
    )
    parser.add_argument(
        "--images_root", 
        required=False,
        default=r"D:\search\fungi\26\data\FungiTastic-Mini\val\300p",
        help="可选：图像目录（用于尺寸对齐）"
    )
    parser.add_argument(
        "--out_report", 
        required=False,
        default=r"D:\search\fungi\26\FungiTastic\out\evaluation_report_yolo_hqsam.txt",
        help="输出完整报告文件路径（txt）"
    )
    parser.add_argument(
        "--out_csv", 
        required=False,
        default=r"D:\search\fungi\26\FungiTastic\out\low_iou_cases.csv",
        help="输出低 IoU 案例 CSV 路径"
    )
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="低 IoU 的阈值")
    parser.add_argument("--resize_method", default="nearest", choices=["nearest","bilinear"], help="调整分辨率时使用的插值")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not pred_dir.exists():
        print(f"错误: 预测目录不存在: {pred_dir}"); sys.exit(1)
        
    print(f"正在从 {args.gt_parquet} 构建 GT 索引...")
    try:
        gt_index = build_gt_index(args.gt_parquet)
    except Exception as e:
        print(f"错误: GT 索引构建失败: {e}"); sys.exit(1)
        
    print(f"GT 索引条目数: {len(gt_index)}")

    pred_files = sorted([p for p in pred_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"找到 {len(pred_files)} 个预测文件.")

    all_results = []
    
    missing_gt = 0
    missing_pred = 0
    decode_fail = 0
    processed = 0

    for p in tqdm(pred_files, desc="IoU Eval"):
        key = normalize_name(p.name)
        file_stem = p.stem
        current_iou = 0.0
        
        # 1. 加载预测掩码
        try:
            pred_img = Image.open(p).convert("L")
            pred_np = np.asarray(pred_img)
        except Exception:
            missing_pred += 1
            all_results.append({"image_name": file_stem, "iou": None, "status": "ERROR_PRED_READ"})
            continue

        # 2. 查找 GT
        if key not in gt_index:
            missing_gt += 1
            all_results.append({"image_name": file_stem, "iou": None, "status": "ERROR_MISSING_GT"})
            continue

        rows = gt_index[key]
        H, W = pred_np.shape # 默认使用预测图尺寸
        
        # 3. 确定目标尺寸 (H, W)
        gt_H, gt_W = rows[0][1], rows[0][2]
        if gt_H > 0 and gt_W > 0:
            H, W = gt_H, gt_W

        if args.images_root:
            # 优先使用图像根目录中的原始图片尺寸 (最准确)
            img_name_stem = p.stem
            img_fp_candidates = [
                Path(args.images_root) / (img_name_stem + ext)
                for ext in [".jpg", ".JPG", ".png", ".jpeg"]
            ]
            for img_fp in img_fp_candidates:
                if img_fp.exists():
                    try:
                        im = Image.open(img_fp)
                        W, H = im.size
                        break
                    except Exception:
                        pass
        
        # 4. 解码并合并 GT 掩码
        combined_gt = np.zeros((H, W), dtype=np.uint8)
        any_decoded = False
        for rle, rh, rw in rows:
            mask_H = rh if rh > 0 else H
            mask_W = rw if rw > 0 else W
            
            mask = rle_decode_pairs_or_counts(rle, mask_H, mask_W)
            
            if mask is not None:
                if mask.shape != (H, W):
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                combined_gt = np.maximum(combined_gt, (mask > 0).astype(np.uint8) * 255)
                any_decoded = True

        if not any_decoded:
            decode_fail += 1
            all_results.append({"image_name": file_stem, "iou": None, "status": "ERROR_GT_DECODE"})
            continue

        # 5. 调整预测掩码尺寸
        if pred_np.shape != (H, W):
            interp = cv2.INTER_NEAREST if args.resize_method == "nearest" else cv2.INTER_LINEAR
            pred_resized = cv2.resize(pred_np, (W, H), interpolation=interp)
        else:
            pred_resized = pred_np

        # 6. 计算 IoU
        current_iou = compute_iou_np(pred_resized, combined_gt)
        processed += 1
        all_results.append({"image_name": file_stem, "iou": current_iou, "status": "OK"})

    # =================================================================
    # 结果报告与输出
    # =================================================================

    df_results = pd.DataFrame(all_results)
    
    # 过滤有效 IoU
    df_valid = df_results.dropna(subset=['iou'])
    ious_arr = df_valid['iou'].to_numpy()

    # 1. 统计报告
    mean_iou = float(np.mean(ious_arr)) if ious_arr.size > 0 else 0.0
    median_iou = float(np.median(ious_arr)) if ious_arr.size > 0 else 0.0

    report = [
        "=== Evaluation Report ===",
        f"Pred dir: {args.pred_dir}",
        f"GT parquet: {args.gt_parquet}",
        f"IoU Threshold: < {args.iou_threshold}",
        f"Pred files scanned: {len(pred_files)}",
        f"Processed (IoU Calculated): {processed}",
        f"Missing GT: {missing_gt}",
        f"Prediction Read Errors: {missing_pred}",
        f"GT Decode Failures: {decode_fail}",
        "",
        f"Mean IoU: {mean_iou:.6f}",
        f"Median IoU: {median_iou:.6f}",
    ]

    outp_report = Path(args.out_report)
    with open(outp_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))
    print(f"完整评估报告已保存到 {outp_report}")
    
    # 2. 低 IoU 案例报告
    df_low_iou = df_valid[df_valid['iou'] < args.iou_threshold].sort_values(by='iou', ascending=True)
    
    outp_csv = Path(args.out_csv)
    df_low_iou.to_csv(outp_csv, index=False, float_format='%.6f')

    print(f"\n=== Low IoU Cases (IoU < {args.iou_threshold}) ===")
    print(f"总计 {len(df_low_iou)} 个案例.")
    print(df_low_iou[['image_name', 'iou']].head(10))
    print(f"低 IoU 案例列表已保存到 {outp_csv}")
    print("Done.")

if __name__ == "__main__":
    main()