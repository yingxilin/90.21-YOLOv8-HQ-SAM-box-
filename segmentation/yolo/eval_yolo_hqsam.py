#!/usr/bin/env python3
"""
Evaluate YOLO+HQ-SAM segmentation masks against ground truth.

Usage:
    python baselines/segmentation/yolo/eval_yolo_hqsam.py \
        --pred_masks "D:/search/fungi/26/FungiTastic/out/masks_yolo_hqsam" \
        --data_root "D:/search/fungi/26/data/FungiTastic-Mini" \
        --split "val" \
        --size "300" \
        --output_dir "D:/search/fungi/26/FungiTastic/out/eval_results"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure project root is in path for imports
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchmetrics.functional import jaccard_index

from dataset.mask_fungi import MaskFungiTastic


def evaluate_single_image(idx, dataset, pred_masks_dir, thresh=0.5, vis=False):
    """Evaluate a single image prediction against ground truth."""
    image, gt_masks, class_id, file_path, label_data = dataset[idx]
    
    # Get predicted mask path
    mask_filename = Path(file_path).stem + ".png"
    pred_mask_path = Path(pred_masks_dir) / mask_filename
    
    if not pred_mask_path.exists():
        print(f"Warning: Prediction not found for {mask_filename}")
        return None
    
    # Load predicted mask
    pred_mask = Image.open(pred_mask_path)
    
    # Resize pred_mask to gt_mask size if needed
    if pred_mask.size != (gt_masks.shape[1], gt_masks.shape[0]):
        pred_mask = pred_mask.resize((gt_masks.shape[1], gt_masks.shape[0]), Image.NEAREST)
    
    # Convert to tensor
    pred_mask_array = np.array(pred_mask)
    
    # Handle different segmentation tasks
    if dataset.seg_task == 'binary':
        gt_mask = gt_masks
        iou = jaccard_index(
            preds=(torch.tensor(pred_mask_array) / 255.0),
            target=torch.tensor(gt_mask),
            task='binary',
            threshold=thresh,
        )
    else:
        raise ValueError(f"Unknown seg_task: {dataset.seg_task}")
    
    if vis:
        # Visualize image, gt_mask and pred_mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title(f"Image\nClass: {dataset.category_id2label.get(class_id, 'Unknown')}")
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title(f"Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask_array, cmap='gray')
        axes[2].set_title(f"Prediction")
        axes[2].axis('off')
        
        plt.suptitle(f"ID: {idx} | File: {Path(file_path).name} | IoU: {iou:.4f}")
        plt.tight_layout()
        plt.show()
    
    return iou.item()


def evaluate_masks(
    dataset,
    pred_masks_dir: str,
    thresh: float = 0.5,
    debug: bool = False,
    vis: bool = False,
    output_dir: Optional[str] = None,
):
    """Evaluate all predicted masks against ground truth."""
    ious = []
    
    # Select indices to evaluate
    if debug:
        idxs = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)
    else:
        idxs = np.arange(len(dataset))
    
    print(f"Evaluating {len(idxs)} images...")
    
    # Evaluate each image
    for idx in tqdm(idxs):
        iou = evaluate_single_image(idx, dataset, pred_masks_dir, thresh, vis)
        if iou is not None:
            ious.append(iou)
    
    if len(ious) == 0:
        print("Error: No valid predictions found!")
        return
    
    # Calculate statistics
    ious = np.array(ious)
    iou_mean = ious.mean()
    iou_std = ious.std()
    iou_median = np.median(ious)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of images evaluated: {len(ious)}")
    print(f"Mean IoU: {iou_mean:.4f}")
    print(f"Std IoU: {iou_std:.4f}")
    print(f"Median IoU: {iou_median:.4f}")
    print(f"Min IoU: {ious.min():.4f}")
    print(f"Max IoU: {ious.max():.4f}")
    print("="*60)
    
    # Save results if output directory specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save IoU values
        np.save(output_path / f'ious_thresh{int(thresh * 100)}.npy', ious)
        
        # Save statistics
        with open(output_path / f'iou_stats_thresh{int(thresh * 100)}.txt', 'w') as f:
            f.write(f"Number of images: {len(ious)}\n")
            f.write(f"Mean IoU: {iou_mean:.4f}\n")
            f.write(f"Std IoU: {iou_std:.4f}\n")
            f.write(f"Median IoU: {iou_median:.4f}\n")
            f.write(f"Min IoU: {ious.min():.4f}\n")
            f.write(f"Max IoU: {ious.max():.4f}\n")
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(ious, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(iou_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {iou_mean:.4f}')
        plt.axvline(iou_mean + iou_std, color='g', linestyle='--', linewidth=1, label=f'Std: ±{iou_std:.4f}')
        plt.axvline(iou_mean - iou_std, color='g', linestyle='--', linewidth=1)
        plt.xlabel('IoU', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'IoU Distribution\nMean: {iou_mean:.4f} ± {iou_std:.4f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'iou_hist_thresh{int(thresh * 100)}.png', dpi=150)
        print(f"\nResults saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO+HQ-SAM segmentation masks")
    parser.add_argument("--pred_masks", required=True, help="Directory containing predicted masks")
    parser.add_argument("--data_root", default="D:/search/fungi/26/data", help="Root directory containing FungiTastic dataset and metadata")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--size", default="300", help="Image size (300 or 600)")
    parser.add_argument("--data_subset", default="Mini", help="Dataset subset (Mini or Full)")
    parser.add_argument("--thresh", type=float, default=0.5, help="Binary threshold for IoU calculation")
    parser.add_argument("--output_dir", help="Directory to save evaluation results")
    parser.add_argument("--debug", action="store_true", help="Debug mode (evaluate only 10 images)")
    parser.add_argument("--vis", action="store_true", help="Visualize predictions")
    
    args = parser.parse_args()
    
    # Check if prediction directory exists
    pred_masks_path = Path(args.pred_masks)
    if not pred_masks_path.exists():
        print(f"Error: Prediction directory not found: {pred_masks_path}")
        sys.exit(1)
    
    # Count predicted masks
    pred_count = len(list(pred_masks_path.glob("*.png")))
    print(f"Found {pred_count} predicted masks in {pred_masks_path}")
    
    if pred_count == 0:
        print("Error: No predicted masks found!")
        sys.exit(1)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    dataset = MaskFungiTastic(
        root=args.data_root,
        split=args.split,
        size=args.size,
        task='closed',
        data_subset=args.data_subset,
        transform=None,
        seg_task='binary',
        debug=False,
    )
    print(f"Dataset loaded: {len(dataset)} images")
    
    # Evaluate
    evaluate_masks(
        dataset=dataset,
        pred_masks_dir=args.pred_masks,
        thresh=args.thresh,
        debug=args.debug,
        vis=args.vis,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
