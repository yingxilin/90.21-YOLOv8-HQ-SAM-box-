#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于原始eval.py的评估脚本
使用MaskFungiTastic数据集类来正确加载GT掩码
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchmetrics.functional import jaccard_index

# 尝试导入数据集类
try:
    # 添加项目根目录到路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(SCRIPT_DIR, '..', '..'))
    sys.path.append(os.path.join(SCRIPT_DIR, '..'))
    sys.path.append(SCRIPT_DIR)
    
    from dataset.mask_fungi import MaskFungiTastic
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False
    print("Warning: Could not import MaskFungiTastic dataset class")


def evaluate_single_image(args):
    """评估单张图像"""
    idx, dataset, pred_dir, thresh, debug_mode, debug_dir = args
    
    try:
        # 从数据集获取GT
        image, gt_masks, class_id, file_path, label_data = dataset[idx]
        
        # 获取对应的预测掩码 - 尝试多种文件名格式
        base_name = os.path.basename(file_path)
        stem = Path(base_name).stem
        
        # 尝试多种可能的文件名
        possible_names = [
            base_name,  # 原始文件名
            f"{stem}.png",  # stem + .png
            base_name.replace('.jpg', '.png'),  # jpg替换为png
            base_name.replace('.JPG', '.png'),  # JPG替换为png
        ]
        
        pred_path = None
        for name in possible_names:
            test_path = os.path.join(pred_dir, name)
            if os.path.exists(test_path):
                pred_path = test_path
                break
        
        if pred_path is None:
            return None, f"Prediction not found for: {base_name}"
        
        # 读取预测掩码
        pred_mask = Image.open(pred_path)
        
        # 关键：将预测resize到GT尺寸（而不是相反）
        pred_mask = pred_mask.resize((gt_masks.shape[1], gt_masks.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_mask)
        
        # 处理二值分割任务
        if dataset.seg_task == 'binary':
            gt_mask = gt_masks
            
            # 使用torchmetrics计算IoU
            iou = jaccard_index(
                preds=(torch.tensor(pred_mask) / 255.0),
                target=torch.tensor(gt_mask),
                task='binary',
                threshold=thresh,
            )
            
            iou_value = iou.item()
        else:
            return None, f"Unsupported seg_task: {dataset.seg_task}"
        
        # 调试可视化
        if debug_mode and debug_dir and np.random.random() < 0.01:
            stem = Path(file_path).stem
            
            # 保存预测和GT
            Image.fromarray((pred_mask).astype(np.uint8)).save(
                debug_dir / f"{stem}_pred.png"
            )
            Image.fromarray((gt_mask * 255).astype(np.uint8)).save(
                debug_dir / f"{stem}_gt.png"
            )
            
            # 创建叠加图
            overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            overlay[:, :, 0] = (gt_mask * 255).astype(np.uint8)  # GT为红色
            overlay[:, :, 1] = pred_mask  # 预测为绿色
            Image.fromarray(overlay).save(debug_dir / f"{stem}_overlay.png")
        
        return iou_value, None
        
    except Exception as e:
        import traceback
        return None, f"Error at index {idx}: {str(e)}\n{traceback.format_exc()}"


def evaluate_saved_masks(
    dataset,
    pred_dir,
    thresh=0.5,
    debug=False,
    workers=8,
    result_dir=None
):
    """评估所有保存的掩码"""
    
    pred_dir = Path(pred_dir)
    
    # 创建调试目录
    debug_dir = None
    if debug:
        debug_dir = Path("debug_visualization")
        debug_dir.mkdir(exist_ok=True)
        print(f"Debug mode enabled. Visualizations will be saved to: {debug_dir}")
    
    # 准备评估任务
    print(f"\nPreparing to evaluate {len(dataset)} images...")
    
    # 首先列出所有预测文件
    pred_files = list(pred_dir.glob("*.png"))
    pred_dict = {p.stem.lower(): p for p in pred_files}
    
    print(f"Found {len(pred_files)} prediction files in {pred_dir}")
    if pred_files:
        print(f"Sample prediction filenames: {[p.name for p in pred_files[:3]]}")
    
    # 检查有多少预测文件存在
    existing_preds = 0
    indices = []
    missing_samples = []
    
    for idx in range(len(dataset)):
        _, _, _, file_path, _ = dataset[idx]
        
        # 尝试多种文件名匹配方式
        base_name = os.path.basename(file_path)
        stem = Path(base_name).stem.lower()
        
        # 尝试直接匹配
        if (pred_dir / base_name).exists():
            existing_preds += 1
            indices.append(idx)
        # 尝试stem匹配（忽略扩展名）
        elif stem in pred_dict:
            existing_preds += 1
            indices.append(idx)
        # 尝试添加.png扩展名
        elif (pred_dir / f"{stem}.png").exists():
            existing_preds += 1
            indices.append(idx)
        else:
            if len(missing_samples) < 5:
                missing_samples.append(base_name)
    
    print(f"Found {existing_preds}/{len(dataset)} matching predictions")
    
    if existing_preds == 0 and missing_samples:
        print(f"\nSample GT filenames that couldn't be matched:")
        for fname in missing_samples:
            print(f"  - {fname}")
    
    if existing_preds == 0:
        print("Error: No matching predictions found!")
        return
    
    # 并行评估
    ious = []
    errors = []
    
    print(f"\nEvaluating with {workers} workers...")
    
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = [
                executor.submit(
                    evaluate_single_image,
                    (idx, dataset, str(pred_dir), thresh, debug, debug_dir)
                )
                for idx in indices
            ]
            
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                iou, error = future.result()
                if error:
                    errors.append(error)
                elif iou is not None:
                    ious.append(iou)
    else:
        # 单线程模式
        for idx in tqdm(indices):
            iou, error = evaluate_single_image(
                (idx, dataset, str(pred_dir), thresh, debug, debug_dir)
            )
            if error:
                errors.append(error)
            elif iou is not None:
                ious.append(iou)
    
    if len(ious) == 0:
        print("\nError: No valid IoU scores computed!")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for err in errors[:10]:
                print(f"  - {err}")
        return
    
    # 计算统计数据
    ious = np.array(ious)
    iou_mean = ious.mean()
    iou_std = ious.std()
    iou_median = np.median(ious)
    
    # 生成报告
    report = f"""
{'='*60}
Segmentation Evaluation Report (Threshold: {thresh})
{'='*60}

Dataset Statistics:
  - Total images in dataset: {len(dataset)}
  - Predictions found: {existing_preds}
  - Successfully evaluated: {len(ious)}
  - Evaluation errors: {len(errors)}

IoU Metrics:
  - Mean IoU:   {iou_mean:.4f}
  - Median IoU: {iou_median:.4f}
  - Std IoU:    {iou_std:.4f}
  - Min IoU:    {ious.min():.4f}
  - Max IoU:    {ious.max():.4f}

IoU Distribution:
  - IoU >= 0.9: {(ious >= 0.9).sum():5d} ({(ious >= 0.9).sum()/len(ious)*100:.1f}%)
  - IoU >= 0.8: {(ious >= 0.8).sum():5d} ({(ious >= 0.8).sum()/len(ious)*100:.1f}%)
  - IoU >= 0.7: {(ious >= 0.7).sum():5d} ({(ious >= 0.7).sum()/len(ious)*100:.1f}%)
  - IoU >= 0.5: {(ious >= 0.5).sum():5d} ({(ious >= 0.5).sum()/len(ious)*100:.1f}%)
  - IoU <  0.5: {(ious < 0.5).sum():5d} ({(ious < 0.5).sum()/len(ious)*100:.1f}%)

Percentiles:
  - 25th: {np.percentile(ious, 25):.4f}
  - 50th: {np.percentile(ious, 50):.4f}
  - 75th: {np.percentile(ious, 75):.4f}
  - 90th: {np.percentile(ious, 90):.4f}
  - 95th: {np.percentile(ious, 95):.4f}

{'='*60}
"""
    
    print(report)
    
    # 保存结果
    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存报告
        report_path = result_dir / f'evaluation_report_thresh{int(thresh*100)}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        # 保存IoU数组
        np.save(result_dir / f'ious_thresh{int(thresh*100)}.npy', ious)
        
        # 保存错误日志
        if errors:
            error_path = result_dir / f'errors_thresh{int(thresh*100)}.txt'
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(errors))
            print(f"Error log saved to: {error_path}")
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(ious, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(iou_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {iou_mean:.3f}')
        plt.axvline(iou_mean + iou_std, color='g', linestyle='--', linewidth=1, label=f'±Std: {iou_std:.3f}')
        plt.axvline(iou_mean - iou_std, color='g', linestyle='--', linewidth=1)
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.title(f'IoU Distribution (Mean: {iou_mean:.3f} ± {iou_std:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        hist_path = result_dir / f'iou_hist_thresh{int(thresh*100)}.png'
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {hist_path}")
        plt.close()
    
    return ious


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks using MaskFungiTastic dataset")
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction masks")
    parser.add_argument("--data_path", required=True, help="Root path of FungiTastic dataset")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--size", default="300", choices=["300", "512", "1024"], help="Image size")
    parser.add_argument("--data_subset", default="Mini", choices=["Mini", "Full"], help="Dataset subset")
    parser.add_argument("--out_dir", required=True, help="Output directory for results")
    parser.add_argument("--thresh", type=float, default=0.5, help="Binary threshold")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    
    args = parser.parse_args()
    
    if not DATASET_AVAILABLE:
        print("Error: MaskFungiTastic dataset class not available!")
        print("Please make sure the dataset module is in the correct path.")
        sys.exit(1)
    
    # 验证路径
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path not found: {data_path}")
        sys.exit(1)
    
    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"Error: Prediction directory not found: {pred_dir}")
        sys.exit(1)
    
    # 加载数据集
    print(f"Loading dataset from: {data_path}")
    print(f"Split: {args.split}, Size: {args.size}p, Subset: {args.data_subset}")
    
    try:
        dataset = MaskFungiTastic(
            root=str(data_path),
            split=args.split,
            size=args.size,
            task='closed',
            data_subset=args.data_subset,
            transform=None,
            seg_task='binary',
            debug=False,
        )
        print(f"Dataset loaded: {len(dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 评估
    evaluate_saved_masks(
        dataset=dataset,
        pred_dir=args.pred_dir,
        thresh=args.thresh,
        debug=args.debug,
        workers=args.workers,
        result_dir=args.out_dir
    )


if __name__ == '__main__':
    main()