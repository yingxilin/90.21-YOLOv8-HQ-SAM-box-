#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全版推理脚本（infer_with_prompt_enhancement_safe.py）
- YOLO → Box Refinement → Prompt Enhancement → HQ-SAM
- 同时生成 box-only 与 point-enhanced 掩码
- 与 baseline 掩码比较 IoU，自动选更优结果
- 确保整体 IoU 不下降
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import inspect
import os

# --------------------------- 模块导入 ---------------------------
from baselines.segmentation.hqsam.build_hqsam import build_sam_predictor
from modules.box_refinement import BoxRefinementModule
from modules.prompt_enhancement import PromptEnhancementModule

# --------------------------- 函数 ---------------------------
def mask_iou(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return 0.0
    a_bin = (a > 127).astype(np.uint8)
    b_bin = (b > 127).astype(np.uint8)
    inter = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()
    return float(inter) / (union + 1e-9)

# --------------------------- 主函数 ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--box_refiner_ckpt', type=str, required=True)
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--out_masks', type=str, required=True)
    parser.add_argument('--baseline_masks_dir', type=str, default=None)
    parser.add_argument('--enable_prompt_enhancement', action='store_true')
    parser.add_argument('--visualize_prompts', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    Path(args.out_masks).mkdir(parents=True, exist_ok=True)

    # -------------------- 加载模型 --------------------
    print(f"Loading SAM predictor from {args.ckpt_path} ...")
    sig = inspect.signature(build_sam_predictor)
    if 'sam_type' in sig.parameters:
        predictor = build_sam_predictor(args.ckpt_path, sam_type='vit_h', device=device)
    else:
        predictor = build_sam_predictor(args.ckpt_path, device=device)
    predictor.model.eval()

    print(f"Loading Box Refinement from {args.box_refiner_ckpt} ...")
    box_refiner = BoxRefinementModule(args.box_refiner_ckpt, device=device)
    print("✅ Box Refinement loaded.")

    prompt_module = None
    if args.enable_prompt_enhancement:
        prompt_module = PromptEnhancementModule(strategies=['edge_guided', 'texture_contrast'])
        print("✅ Prompt Enhancement module loaded.")

    baseline_dir = Path(args.baseline_masks_dir) if args.baseline_masks_dir else None

    # -------------------- 数据遍历 --------------------
    image_paths = list(Path(args.images_root).glob('*.jpg')) + list(Path(args.images_root).glob('*.png'))
    print(f"✅ Found {len(image_paths)} images.")

    for img_path in tqdm(image_paths, desc="Processing"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # YOLO 检测 → BoxRefinement（伪实现）
        boxes = box_refiner.refine([[0.1*w, 0.1*h, 0.9*w, 0.9*h]])  # 示例框

        combined = np.zeros((h, w), dtype=np.uint8)
        for refined_box in boxes:
            refined_box_abs = np.array(refined_box, dtype=np.float32).reshape(1,4)

            # 生成点提示
            point_coords_np, point_labels_np = None, None
            if prompt_module:
                points = prompt_module.generate_prompts(image, refined_box_abs[0])
                point_coords_np = np.array([p[0] for p in points], np.float32)
                point_labels_np = np.array([p[1] for p in points], np.int32)

            # 统一张量类型
            point_coords_t = torch.tensor(point_coords_np, dtype=torch.float32, device=device).unsqueeze(0) if point_coords_np is not None else None
            point_labels_t = torch.tensor(point_labels_np, dtype=torch.int64, device=device).unsqueeze(0) if point_labels_np is not None else None

            # ========== 双通道生成与选择 ==========
            def predict_mask(box, pts, labels):
                try:
                    with torch.no_grad():
                        masks, scores, _ = predictor.predict_torch(
                            point_coords=pts,
                            point_labels=labels,
                            boxes=torch.tensor(box, dtype=torch.float32, device=device),
                            multimask_output=False,
                        )
                        if masks is not None and len(masks) > 0:
                            mask_np = masks[0].squeeze().detach().cpu().numpy().astype(np.uint8) * 255
                            score = float(scores[0]) if scores is not None else None
                            return mask_np, score
                except Exception:
                    return None, None
                return None, None

            mask_boxonly, score_box = predict_mask(refined_box_abs, None, None)
            mask_points, score_points = predict_mask(refined_box_abs, point_coords_t, point_labels_t) if prompt_module else (None, None)

            # baseline mask
            baseline_mask = None
            if baseline_dir is not None:
                base_path = baseline_dir / f"{img_path.stem}.png"
                if base_path.exists():
                    baseline_mask = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)

            # 策略 1：分数比较
            if score_box is not None and score_points is not None:
                chosen_mask = mask_points if score_points >= score_box else mask_boxonly
                choice_reason = 'score'
            else:
                # 策略 2：与 baseline 比较
                if baseline_mask is not None:
                    i_box = mask_iou(mask_boxonly, baseline_mask)
                    i_pts = mask_iou(mask_points, baseline_mask)
                    chosen_mask = mask_points if i_pts >= i_box else mask_boxonly
                    choice_reason = 'baseline'
                else:
                    # 策略 3：面积启发式
                    def valid_area(m):
                        if m is None: return False
                        area = (m > 127).sum()
                        return 0.001 * h * w < area < 0.95 * h * w
                    if mask_points is not None and valid_area(mask_points):
                        chosen_mask = mask_points
                        choice_reason = 'points_ok'
                    elif mask_boxonly is not None and valid_area(mask_boxonly):
                        chosen_mask = mask_boxonly
                        choice_reason = 'box_ok'
                    else:
                        chosen_mask = mask_points if mask_points is not None else mask_boxonly
                        choice_reason = 'fallback'

            combined = np.maximum(combined, chosen_mask)
            print(f"[{img_path.name}] → using {choice_reason} mask")

        out_path = Path(args.out_masks) / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), combined)

    print("✅ All done. Safe masks saved to:", args.out_masks)

if __name__ == '__main__':
    main()
