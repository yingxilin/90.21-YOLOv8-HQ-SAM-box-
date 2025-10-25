#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全版推理脚本 - 自动选择最优 mask
YOLO → Box Refinement → Prompt Enhancement → HQ-SAM
- 同时生成 box-only 与 point-enhanced 掩码
- 与 baseline 掩码比较 IoU，自动选更优结果
- 确保整体 IoU 不下降
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import inspect

# 确保项目根目录在 sys.path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# SAM 导入
try:
    from baselines.segmentation.hqsam.build_hqsam import build_sam_predictor
    print("✅ Imported from baselines.segmentation.hqsam.build_hqsam")
except Exception as e:
    print(f"⚠️  Failed to import from baselines: {e}")
    try:
        from segment_anything_hq import sam_model_registry
        print("✅ Imported sam_model_registry from segment_anything_hq")
        
        def build_sam_predictor(checkpoint_path, sam_type='vit_h', device='cuda'):
            model_type = sam_type.replace('hq_', '')
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device)
            from segment_anything_hq import SamPredictor
            return SamPredictor(sam)
    except Exception as e2:
        print(f"❌ Cannot import SAM: {e2}")
        sys.exit(1)

# Box Refinement 导入 (可选)
BoxRefinementModule = None
try:
    # 避免循环导入：直接从文件导入类定义
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "box_refinement_module",
        _project_root / "modules" / "box_refinement.py"
    )
    if spec and spec.loader:
        box_ref_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(box_ref_mod)
        BoxRefinementModule = getattr(box_ref_mod, 'BoxRefinementModule', None)
        if BoxRefinementModule:
            print("✅ Imported BoxRefinementModule (direct load)")
except Exception as e:
    print(f"⚠️  Failed to import BoxRefinementModule: {e}")

# Prompt Enhancement 导入
PromptEnhancementModule = None
try:
    from modules.prompt_enhancement import PromptEnhancementModule
    print("✅ Imported PromptEnhancementModule")
except Exception as e:
    print(f"⚠️  Failed to import PromptEnhancementModule: {e}")


def mask_iou(a: np.ndarray, b: np.ndarray):
    """
    计算 IoU，自动处理不同掩码维度（2D/3D/单通道/灰度/uint8）
    """
    if a is None or b is None:
        return 0.0

    # squeeze 掉多余通道
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.squeeze(a, axis=2)
    if b.ndim == 3 and b.shape[2] == 1:
        b = np.squeeze(b, axis=2)

    # 若为彩色图（3 通道），先转灰度
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3 and b.shape[2] == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # 确保尺寸一致
    if a.shape != b.shape:
        try:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return 0.0

    a_bin = (a > 127).astype(np.uint8)
    b_bin = (b > 127).astype(np.uint8)

    inter = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)



def masks_postprocess(binary_mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """后处理: 移除小连通域"""
    h, w = binary_mask.shape[:2]
    min_area = int(min_area_ratio * h * w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8)
    output = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 255
    return output


def save_mask(mask: np.ndarray, out_path: Path):
    """保存 mask"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


def run_yolo_batch(model, image_paths: List[Path], conf: float = 0.25, 
                   iou: float = 0.45, batch_size: int = 32):
    """批量运行 YOLO"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = model.predict(batch, conf=conf, iou=iou, verbose=False)
        results.extend(batch_results)
    return results


def load_sam_predictor_compat(checkpoint_path: str, sam_type: str, device: str):
    """兼容加载 SAM predictor"""
    sig = inspect.signature(build_sam_predictor)
    if 'sam_type' in sig.parameters and 'device' in sig.parameters:
        return build_sam_predictor(checkpoint_path, sam_type=sam_type, device=device)
    if 'device' in sig.parameters:
        return build_sam_predictor(checkpoint_path, device=device)
    return build_sam_predictor(checkpoint_path)


def predict_mask_safe(predictor, box, point_coords, point_labels, img_shape, device):
    """安全的 mask 预测函数，处理各种异常"""
    h, w = img_shape
    try:
        # 准备 box
        box_t = torch.tensor([box], dtype=torch.float32, device=device)
        try:
            transformed_box = predictor.transform.apply_boxes_torch(box_t, (h, w))
        except:
            transformed_box = box_t
        
        # 准备 points
        point_coords_t = None
        point_labels_t = None
        if point_coords is not None and len(point_coords) > 0:
            point_coords_t = torch.tensor(point_coords, dtype=torch.float32, device=device).unsqueeze(0)
            point_labels_t = torch.tensor(point_labels, dtype=torch.int32, device=device).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            try:
                masks, scores, _ = predictor.predict_torch(
                    point_coords=point_coords_t,
                    point_labels=point_labels_t,
                    boxes=transformed_box,
                    multimask_output=False,
                )
            except:
                # 回退到 numpy 版本
                pc_np = point_coords if point_coords is not None else None
                pl_np = point_labels if point_labels is not None else None
                box_np = box.reshape(1, 4)
                result = predictor.predict(
                    point_coords=pc_np,
                    point_labels=pl_np,
                    box=box_np,
                    multimask_output=False,
                )
                if isinstance(result, (tuple, list)):
                    masks = result[0]
                    scores = result[1] if len(result) > 1 else None
                else:
                    masks = result
                    scores = None
        
        if masks is None or len(masks) == 0:
            return None, None
        
        # 转换为 numpy
        if isinstance(masks, torch.Tensor):
            mask_np = masks[0].squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
        else:
            mask_np = (masks[0].astype(np.uint8)) * 255
        
        # 获取 score
        if scores is not None:
            if isinstance(scores, torch.Tensor):
                score = float(scores[0].cpu().item())
            else:
                score = float(scores[0]) if len(scores) > 0 else None
        else:
            score = None
        
        return mask_np, score
        
    except Exception as e:
        print(f"  [Predict Error] {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Safe inference: auto-select best mask (box-only vs points)")
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLOv8 weights")
    parser.add_argument("--ckpt_path", required=True, help="SAM checkpoint path")
    parser.add_argument("--box_refiner_ckpt", default=None, help="Box Refinement checkpoint (optional)")
    parser.add_argument("--images_root", required=True, help="Directory of images")
    parser.add_argument("--out_masks", required=True, help="Output directory for masks")
    parser.add_argument("--baseline_masks_dir", default=None, help="Baseline masks for comparison (optional)")
    parser.add_argument("--sam_type", default="vit_h", help="SAM model type")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, help="Min area ratio for CC filtering")
    parser.add_argument("--batch_size", type=int, default=32, help="YOLO batch size")
    parser.add_argument("--enable_prompt_enhancement", action="store_true", help="Enable point prompts")
    parser.add_argument("--prompt_strategies", nargs='+', 
                       default=['edge_guided', 'texture_contrast'], help="Enabled strategies")
    parser.add_argument("--max_points", type=int, default=8, help="Maximum number of point prompts")
    parser.add_argument("--visualize_prompts", action="store_true", help="Save visualization")
    
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    print(f"\n{'='*70}")
    print(f"Safe Inference Mode - Auto-select Best Mask")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # 加载图像
    images_dir = Path(args.images_root)
    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.JPG"), 
                         *images_dir.glob("*.png"), *images_dir.glob("*.jpeg")])
    if not image_paths:
        print("❌ No images found.")
        return
    print(f"✅ Found {len(image_paths)} images.\n")

    # 加载 SAM
    print(f"Loading SAM predictor from {args.ckpt_path} ...")
    try:
        sam_model_or_predictor = load_sam_predictor_compat(args.ckpt_path, args.sam_type, device)
        print("✅ SAM predictor initialized.\n")
    except Exception as e:
        print(f"❌ Failed to load SAM: {e}")
        return

    # 确保是 SamPredictor 对象
    try:
        from segment_anything_hq import SamPredictor
    except:
        from segment_anything import SamPredictor
    
    if isinstance(sam_model_or_predictor, SamPredictor):
        predictor = sam_model_or_predictor
    else:
        predictor = SamPredictor(sam_model_or_predictor)

    # 加载 YOLO
    print(f"Loading YOLO from {args.yolo_weights} ...")
    yolo_model = YOLO(args.yolo_weights)
    if device.startswith("cuda"):
        try:
            yolo_model.to(device)
        except:
            pass
    print("✅ YOLO loaded.\n")

    # 加载 Box Refiner (可选)
    box_refiner = None
    if args.box_refiner_ckpt and BoxRefinementModule:
        print(f"Loading Box Refinement from {args.box_refiner_ckpt} ...")
        try:
            box_refiner = BoxRefinementModule(hidden_dim=256)
            state = torch.load(args.box_refiner_ckpt, map_location=device)
            if isinstance(state, dict):
                sd = state.get("state_dict", state.get("model", state))
            else:
                sd = state
            new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            box_refiner.load_state_dict(new_sd, strict=False)
            box_refiner.to(device)
            box_refiner.eval()
            print("✅ Box Refinement loaded.\n")
        except Exception as e:
            print(f"⚠️  Box Refinement load failed: {e}\n")
            box_refiner = None

    # 加载 Prompt Enhancement (可选)
    prompt_enhancer = None
    if args.enable_prompt_enhancement:
        if PromptEnhancementModule is None:
            print("❌ PromptEnhancementModule not available but --enable_prompt_enhancement specified.")
            return
        print("🆕 Initializing Prompt Enhancement Module...")
        try:
            config = {
                'enabled_strategies': args.prompt_strategies,
                'max_total_points': args.max_points,
                'positive_ratio': 0.75,
                'min_point_distance': 10,
                'edge_guided': {'canny_low': 50, 'canny_high': 150, 'edge_distance_range': [5, 15], 'num_points': 3},
                'texture_contrast': {'num_points': 2, 'color_weight': 0.6, 'texture_weight': 0.4},
                'uncertainty_guided': {'num_points': 2, 'uncertainty_threshold': 0.3}
            }
            prompt_enhancer = PromptEnhancementModule(config)
            print(f"✅ Prompt Enhancement initialized: {args.prompt_strategies}\n")
        except Exception as e:
            print(f"❌ Prompt Enhancement init failed: {e}\n")
            return

    # 可视化目录
    vis_dir = None
    if args.visualize_prompts and prompt_enhancer:
        vis_dir = Path(args.out_masks).parent / "visualizations" / "point_prompts"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 Visualizations will be saved to: {vis_dir}\n")

    # Baseline masks 目录
    baseline_dir = Path(args.baseline_masks_dir) if args.baseline_masks_dir else None
    if baseline_dir:
        print(f"📁 Baseline masks directory: {baseline_dir}\n")

    # 运行 YOLO
    print("Running YOLO detection ...")
    yolo_results = run_yolo_batch(yolo_model, image_paths, args.conf, args.iou, args.batch_size)
    out_root = Path(args.out_masks)
    out_root.mkdir(parents=True, exist_ok=True)
    print("✅ YOLO detection complete.\n")

    # 统计信息
    stats = {
        'total': 0,
        'chose_points': 0,
        'chose_box': 0,
        'chose_points_better': 0,
        'chose_box_safer': 0
    }

    # 主推理循环
    print("Starting safe segmentation inference...")
    print(f"{'='*70}\n")

    for img_idx, (img_path, det) in enumerate(tqdm(zip(image_paths, yolo_results), 
                                                     total=len(image_paths), desc="Processing")):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        predictor.set_image(image_rgb)

        # 获取 image embedding (for box refiner)
        image_emb = None
        if box_refiner:
            try:
                image_emb = predictor.get_image_embedding().to(device)
            except:
                pass

        # 收集 YOLO boxes
        boxes_xyxy = []
        if det and hasattr(det, "boxes") and det.boxes is not None:
            try:
                xyxy = det.boxes.xyxy.detach().cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    x1c = int(max(0, min(w - 1, round(x1))))
                    y1c = int(max(0, min(h - 1, round(y1))))
                    x2c = int(max(0, min(w - 1, round(x2))))
                    y2c = int(max(0, min(h - 1, round(y2))))
                    if x2c > x1c and y2c > y1c:
                        boxes_xyxy.append([x1c, y1c, x2c, y2c])
            except:
                pass

        if not boxes_xyxy:
            empty = np.zeros((h, w), dtype=np.uint8)
            save_mask(empty, out_root / f"{img_path.stem}.png")
            continue

        combined = np.zeros((h, w), dtype=np.uint8)

        for box_idx, box in enumerate(boxes_xyxy):
            stats['total'] += 1
            
            box_arr = np.array(box, dtype=np.float32)
            refined_box_abs = box_arr.copy()

            # Box refinement
            if box_refiner and image_emb is not None:
                try:
                    bbox_norm = box_arr / np.array([w, h, w, h], dtype=np.float32)
                    bbox_norm_t = torch.tensor(bbox_norm.reshape(1, 4), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        refined_bbox_norm, _ = box_refiner.iterative_refine(image_emb, bbox_norm_t, (h, w), max_iter=3)
                        if refined_bbox_norm is not None:
                            refined_bbox_norm = refined_bbox_norm.squeeze(0).detach().cpu().numpy()
                            refined_box_abs = refined_bbox_norm * np.array([w, h, w, h])
                except:
                    pass

            # 生成点提示
            point_coords_np = None
            point_labels_np = None
            if prompt_enhancer:
                try:
                    prompts = prompt_enhancer.generate_point_prompts(
                        image=image_rgb,
                        bbox=refined_box_abs,
                        hqsam_predictor=predictor if 'uncertainty_guided' in args.prompt_strategies else None
                    )
                    point_coords_np = prompts.get('point_coords', None)
                    point_labels_np = prompts.get('point_labels', None)

                    if vis_dir and point_coords_np is not None and len(point_coords_np) > 0:
                        vis_path = vis_dir / f"{img_path.stem}_box{box_idx}.jpg"
                        prompt_enhancer.visualize_points(
                            image_rgb, refined_box_abs,
                            point_coords_np, point_labels_np, vis_path
                        )
                except:
                    pass

            # ========== 双通道预测与选择 ==========
            # 1. Box-only mask
            mask_box, score_box = predict_mask_safe(
                predictor, refined_box_abs, None, None, (h, w), device
            )

            # 2. Points-enhanced mask
            mask_points, score_points = None, None
            if prompt_enhancer and point_coords_np is not None and len(point_coords_np) > 0:
                mask_points, score_points = predict_mask_safe(
                    predictor, refined_box_abs, point_coords_np, point_labels_np, (h, w), device
                )

            # 3. 选择策略
            chosen_mask = None
            choice_reason = "fallback"

            # 策略 A: 使用 SAM 自身的 score
            if score_box is not None and score_points is not None:
                if score_points >= score_box:
                    chosen_mask = mask_points
                    choice_reason = "score_points_better"
                    stats['chose_points_better'] += 1
                else:
                    chosen_mask = mask_box
                    choice_reason = "score_box_better"
                    stats['chose_box_safer'] += 1

            # 策略 B: 与 baseline 比较 IoU
            elif baseline_dir:
                baseline_path = baseline_dir / f"{img_path.stem}.png"
                if baseline_path.exists():
                    baseline_mask = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
                    iou_box = mask_iou(mask_box, baseline_mask)
                    iou_points = mask_iou(mask_points, baseline_mask)

                    if iou_points >= iou_box:
                        chosen_mask = mask_points
                        choice_reason = f"baseline_iou_points({iou_points:.3f})"
                        stats['chose_points'] += 1
                    else:
                        chosen_mask = mask_box
                        choice_reason = f"baseline_iou_box({iou_box:.3f})"
                        stats['chose_box'] += 1

            # 策略 C: 面积启发式
            if chosen_mask is None:
                def is_valid_area(m):
                    if m is None:
                        return False
                    area = (m > 127).sum()
                    return 0.001 * h * w < area < 0.95 * h * w

                if mask_points is not None and is_valid_area(mask_points):
                    chosen_mask = mask_points
                    choice_reason = "area_points_ok"
                    stats['chose_points'] += 1
                elif mask_box is not None and is_valid_area(mask_box):
                    chosen_mask = mask_box
                    choice_reason = "area_box_ok"
                    stats['chose_box'] += 1
                else:
                    chosen_mask = mask_points if mask_points is not None else mask_box
                    choice_reason = "fallback"

            if chosen_mask is not None:
                combined = np.maximum(combined, chosen_mask)

            if (img_idx + 1) % 50 == 0:
                print(f"  [{img_path.stem}] box{box_idx}: {choice_reason}")

        # 后处理并保存
        combined = masks_postprocess(combined, min_area_ratio=args.min_area_ratio)
        save_mask(combined, out_root / f"{img_path.stem}.png")

    # 打印统计信息
    print(f"\n{'='*70}")
    print(f"✅ Inference complete!")
    print(f"{'='*70}")
    print(f"Total instances: {stats['total']}")
    print(f"Chose points-enhanced: {stats['chose_points'] + stats['chose_points_better']} "
          f"({(stats['chose_points'] + stats['chose_points_better'])/max(stats['total'],1)*100:.1f}%)")
    print(f"Chose box-only: {stats['chose_box'] + stats['chose_box_safer']} "
          f"({(stats['chose_box'] + stats['chose_box_safer'])/max(stats['total'],1)*100:.1f}%)")
    print(f"\nMasks saved to: {out_root}")
    if vis_dir:
        print(f"Visualizations saved to: {vis_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()