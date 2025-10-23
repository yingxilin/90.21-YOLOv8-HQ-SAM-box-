#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 -> BoxRefinement -> HQ-SAM 推理脚本（整合版）
使用说明（示例，Windows PowerShell）:
python infer_yolo_hqsam_with_boxrefine.py `
  --yolo_weights "D:\search\fungi\26\FungiTastic\runs\detect\fungi_detection\weights\best.pt" `
  --ckpt_path "D:\search\fungi\26\data\models\fungitastic_ckpts" `
  --box_refiner_ckpt "D:\search\fungi\26_2\26\checkpoints\box_refinement\best_model.pth" `
  --images_root "D:\search\fungi\26\data\FungiTastic-Mini\val\300p" `
  --out_masks "D:\search\fungi\26_2\26\gaijinout\masks_with_refine" `
  --sam_type vit_h --conf 0.35 --iou 0.6 --device cuda
说明:
- 我默认保留与你之前成功脚本一致的 SAM 构建方式（通过 baselines.segmentation.hqsam.build_hqsam.build_sam_predictor）。
- box_refiner 模块基于你确认的路径 modules/box_refinement.py 中的 BoxRefinementModule 类。
- 若 box_refiner 失败，脚本将使用原始 YOLO 框继续生成 mask（不会中断）。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# 将项目根加入 sys.path （便于导入 baselines.*）
_this_file = Path(__file__).resolve()
_project_root = _this_file.parents[2]  # 调整层级以匹配你的项目结构；若需要改，请调整
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

# 导入你现有的 build_sam_predictor（与之前成功脚本一致）
try:
    from baselines.segmentation.hqsam.build_hqsam import build_sam_predictor
except Exception:
    # 备用导入路径（防止层级不同）
    from baselines.segmentation.hqsam.build_hqsam import build_sam_predictor

# 导入 BoxRefinement
from modules.box_refinement import BoxRefinementModule

def masks_postprocess(binary_mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    h, w = binary_mask.shape[:2]
    min_area = int(min_area_ratio * h * w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    output = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 255
    return output

def save_mask(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)

def run_yolo_batch(model, image_paths: List[Path], conf: float = 0.25, iou: float = 0.45, batch_size: int = 32):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = model.predict(batch, conf=conf, iou=iou, verbose=False)
        results.extend(batch_results)
    return results

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 -> BoxRefinement -> HQ-SAM inference (with fallback)")
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLOv8 weights .pt")
    parser.add_argument("--ckpt_path", required=True, help="SAM/HQ-SAM checkpoint path or directory")
    parser.add_argument("--box_refiner_ckpt", required=True, help="Box Refinement .pth checkpoint")
    parser.add_argument("--images_root", required=True, help="Directory of images for inference")
    parser.add_argument("--out_masks", required=True, help="Output directory for masks")
    parser.add_argument("--sam_type", default="hq_vit_h", choices=["hq_vit_h", "hq_vit_l", "vit_h", "vit_l"], help="SAM model type")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, help="Minimum area ratio for CC filtering")
    parser.add_argument("--batch_size", type=int, default=32, help="YOLO batch size")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    print(f"Using device: {device}", flush=True)

    images_dir = Path(args.images_root)
    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.JPG"), *images_dir.glob("*.png"), *images_dir.glob("*.jpeg")])
    if not image_paths:
        print("No images found.", flush=True)
        return
    print(f"Found {len(image_paths)} images.", flush=True)

    # ---- load SAM predictor (same as你的成功脚本) ----
    print(f"Loading SAM predictor from {args.ckpt_path} ...", flush=True)
    import inspect
    sig = inspect.signature(build_sam_predictor)
    if "sam_type" in sig.parameters:
        sam_model = build_sam_predictor(args.ckpt_path, sam_type=args.sam_type, device=device)
    else:
        sam_model = build_sam_predictor(args.ckpt_path, device=device)

    # 🧩 手动封装成 SamPredictor（保证有 set_image / predict_torch）
    try:
        from segment_anything_hq import SamPredictor
    except ImportError:
        from segment_anything import SamPredictor

    predictor = SamPredictor(sam_model)
    print("✅ SAM predictor fully initialized (with set_image & predict_torch).", flush=True)


    # ---- load YOLO ----
    print(f"Loading YOLO from {args.yolo_weights} ...", flush=True)
    yolo_model = YOLO(args.yolo_weights)
    if device and device.startswith("cuda"):
        try:
            yolo_model.to(device)
        except Exception:
            pass

    # ---- load BoxRefinement module ----
    print(f"Loading Box Refinement module from {args.box_refiner_ckpt} ...", flush=True)
    box_refiner = BoxRefinementModule(hidden_dim=256)  # hidden_dim 确认与你训练时保持一致
    state = torch.load(args.box_refiner_ckpt, map_location=device)
    # 如果你保存的是 state_dict 或者其他包装，下面这个尽量通用
    if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("layer") for k in state.keys()):
        # try common patterns: either store directly or under 'state_dict'
        sd = state.get("state_dict", state)
    else:
        sd = state
    try:
        box_refiner.load_state_dict(sd)
    except RuntimeError:
        # 有时 checkpoint 会有 module. 前缀
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[nk] = v
        box_refiner.load_state_dict(new_sd)
    box_refiner.to(device)
    box_refiner.eval()
    print("Box Refinement module loaded.", flush=True)

    # ---- run YOLO in batches ----
    print("Running YOLO detection ...", flush=True)
    yolo_results = run_yolo_batch(yolo_model, image_paths, conf=args.conf, iou=args.iou, batch_size=args.batch_size)
    out_root = Path(args.out_masks)
    out_root.mkdir(parents=True, exist_ok=True)

    # loop
    for img_path, det in zip(image_paths, yolo_results):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Failed to read {img_path}", flush=True)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # set image for SAM and get embedding (for box_refiner)
        predictor.set_image(image_rgb)
        # image embedding may be used by the box refiner
        try:
            image_emb = predictor.get_image_embedding().to(device)
        except Exception:
            image_emb = None

        # collect YOLO boxes (absolute xyxy)
        boxes_xyxy = []
        if det is not None and hasattr(det, "boxes") and det.boxes is not None:
            try:
                xyxy = det.boxes.xyxy.detach().cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    x1c = int(max(0, min(w - 1, round(x1))))
                    y1c = int(max(0, min(h - 1, round(y1))))
                    x2c = int(max(0, min(w - 1, round(x2))))
                    y2c = int(max(0, min(h - 1, round(y2))))
                    if x2c > x1c and y2c > y1c:
                        boxes_xyxy.append([x1c, y1c, x2c, y2c])
            except Exception:
                pass

        if not boxes_xyxy:
            # save empty mask
            empty = np.zeros((h, w), dtype=np.uint8)
            save_mask(empty, out_root / f"{img_path.stem}.png")
            continue

        # 将所有 boxes 逐个 refine 并用 SAM 分割（或合并后再后处理）
        combined = np.zeros((h, w), dtype=np.uint8)
        for box in boxes_xyxy:
            try:
                # prepare for refiner: normalized xyxy in [0,1]
                box_arr = np.array(box, dtype=np.float32)  # [x1,y1,x2,y2] absolute
                bbox_norm = box_arr / np.array([w, h, w, h], dtype=np.float32)  # normalized
                bbox_norm_t = torch.tensor(bbox_norm.reshape(1, 4), dtype=torch.float32, device=device)

                refined_box_abs = None
                # run refinement if possible
                if image_emb is not None:
                    with torch.no_grad():
                        # box_refiner.iterative_refine 返回 (refined_bbox, extras) 按你之前实现
                        try:
                            refined_bbox_norm, _ = box_refiner.iterative_refine(image_emb, bbox_norm_t, (h, w), max_iter=3)
                            if refined_bbox_norm is not None:
                                refined_bbox_norm = refined_bbox_norm.squeeze(0).detach().cpu().numpy()
                                refined_box_abs = refined_bbox_norm * np.array([w, h, w, h], dtype=np.float32)
                        except Exception as e:
                            # refinement 出错 -> fallback
                            refined_box_abs = None

                if refined_box_abs is None:
                    # fallback to original YOLO box
                    refined_box_abs = box_arr

                # convert to torch and apply predictor.transform
                refined_boxes_t = torch.tensor([refined_box_abs], dtype=torch.float32, device=device)
                transformed = predictor.transform.apply_boxes_torch(refined_boxes_t, (h, w))

                # predict mask with SAM
                with torch.no_grad():
                    masks, scores, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed,
                        multimask_output=False,
                    )

                if masks is None or len(masks) == 0:
                    continue

                m_np = masks[0].squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
                combined = np.maximum(combined, m_np)

            except Exception as e:
                # 某个实例失败则忽略，继续下一个
                print(f"Warning: instance failed for {img_path.name}: {e}", flush=True)
                continue

        # postprocess & save
        combined = masks_postprocess(combined, min_area_ratio=args.min_area_ratio)
        save_mask(combined, out_root / f"{img_path.stem}.png")

    print(f"Inference done. Masks saved to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
