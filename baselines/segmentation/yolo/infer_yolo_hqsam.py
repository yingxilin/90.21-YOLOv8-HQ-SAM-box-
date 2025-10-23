#!/usr/bin/env python3
"""
Run YOLOv8 detection to get boxes, then refine to masks with SAM or HQ-SAM.

Inputs:
  - YOLO weights path (ultralytics format)
  - SAM/HQ-SAM checkpoint path or directory
  - Images root (a directory of images)
Outputs:
  - Per-image mask PNGs saved under out_masks

Usage example (Windows PowerShell):
  python baselines/segmentation/yolo/infer_yolo_hqsam.py `
    --yolo_weights runs/detect/fungi_detection/weights/best.pt `
    --ckpt_path D:/search/fungi/26/data/models/fungitastic_ckpts `
    --images_root D:/search/fungi/26/data/FungiTastic-Mini/val/300p `
    --out_masks D:/search/fungi/26/FungiTastic/out/yolo_hqsam_masks `
    --sam_type hq_vit_h --conf 0.35 --iou 0.6
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in path for imports
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from baselines.segmentation.hqsam.build_hqsam import build_sam_predictor


def xywh_norm_to_xyxy_abs(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x_center = xc * img_w
    y_center = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(x_center - bw / 2))
    y1 = int(round(y_center - bh / 2))
    x2 = int(round(x_center + bw / 2))
    y2 = int(round(y_center + bh / 2))
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


def run_yolo_batch(model, image_paths: List[Path], conf: float = 0.25, iou: float = 0.45, batch_size: int = 32):
    """Run YOLO in batches to avoid file handle issues."""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = model.predict(batch, conf=conf, iou=iou, verbose=False)
        results.extend(batch_results)
    return results


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


def main():
    print("Starting YOLOv8-det → HQ-SAM inference...", flush=True)
    parser = argparse.ArgumentParser(description="YOLOv8-det → HQ-SAM inference")
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLOv8 weights .pt")
    parser.add_argument("--ckpt_path", required=True, help="SAM/HQ-SAM checkpoint path or directory")
    parser.add_argument("--images_root", required=True, help="Directory of images for inference")
    parser.add_argument("--out_masks", required=True, help="Output directory for masks")
    parser.add_argument("--sam_type", default="hq_vit_h", choices=["hq_vit_h", "hq_vit_l", "vit_h", "vit_l"], help="SAM model type")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, help="Minimum area ratio for CC filtering")
    parser.add_argument("--save_individual", action="store_true", help="Save one mask per image")

    args = parser.parse_args()
    print(f"Arguments parsed successfully", flush=True)

    images_dir = Path(args.images_root)
    if not images_dir.exists():
        print(f"Error: images_root not found: {images_dir}")
        sys.exit(1)

    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.JPG"), *images_dir.glob("*.png")])
    if not image_paths:
        print(f"No images found under {images_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images", flush=True)

    # Load SAM/HQ-SAM predictor
    print(f"Loading HQ-SAM predictor ({args.sam_type})...", flush=True)
    predictor = build_sam_predictor(args.ckpt_path, sam_type=args.sam_type, device=args.device)
    print("HQ-SAM predictor loaded successfully", flush=True)

    # YOLO detection
    print(f"Loading YOLO model from {args.yolo_weights}...", flush=True)
    yolo_model = YOLO(args.yolo_weights)
    if args.device:
        yolo_model.to(args.device)
    print(f"Running YOLO detection on {len(image_paths)} images...", flush=True)
    yolo_results = run_yolo_batch(yolo_model, image_paths, conf=args.conf, iou=args.iou, batch_size=32)

    out_root = Path(args.out_masks)
    out_root.mkdir(parents=True, exist_ok=True)

    for img_path, det in zip(image_paths, yolo_results):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Warning: failed to read {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Set image for SAM
        predictor.set_image(image_rgb)

        # Collect boxes from YOLO
        boxes_xyxy = []
        if det is not None and hasattr(det, "boxes") and det.boxes is not None:
            # Use absolute xyxy from ultralytics results directly when available
            try:
                xyxy = det.boxes.xyxy.detach().cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    x1c = int(max(0, min(w - 1, x1)))
                    y1c = int(max(0, min(h - 1, y1)))
                    x2c = int(max(0, min(w - 1, x2)))
                    y2c = int(max(0, min(h - 1, y2)))
                    if x2c > x1c and y2c > y1c:
                        boxes_xyxy.append([x1c, y1c, x2c, y2c])
            except Exception:
                pass

        if not boxes_xyxy:
            # Nothing detected → save empty mask
            empty = np.zeros((h, w), dtype=np.uint8)
            save_mask(empty, out_root / f"{img_path.stem}.png")
            continue

        # SAM expects boxes as torch tensor in XYXY format
        boxes_t = torch.tensor(boxes_xyxy, device=args.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_t, (h, w))

        with torch.no_grad():
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        # Combine instance masks
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            m_np = m.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
            combined = np.maximum(combined, m_np)

        combined = masks_postprocess(combined, min_area_ratio=args.min_area_ratio)
        save_path = out_root / f"{img_path.stem}.png"
        save_mask(combined, save_path)
        
    print(f"Inference done. Masks saved to: {out_root}")


if __name__ == "__main__":
    main()


