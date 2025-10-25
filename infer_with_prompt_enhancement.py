#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 -> BoxRefinement -> Prompt Enhancement -> HQ-SAM 推理脚本
修正版 - 自动兼容旧版 build_sam_predictor 签名，完整可运行版
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import inspect

# Ensure project root on sys.path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

print(f"Project root: {_project_root}")
print(f"Python path: {sys.path[:3]}")

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

# Try multiple SAM imports / fallbacks
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
        print(f"❌ Cannot import SAM-related modules: {e2}")
        sys.exit(1)

# BoxRefinement import (optional)
try:
    from modules.box_refinement import BoxRefinementModule
    print("✅ Imported BoxRefinementModule")
except Exception as e:
    print(f"⚠️  Failed to import BoxRefinementModule: {e}")
    BoxRefinementModule = None

# Prompt Enhancement import (required when enabled)
try:
    from modules.prompt_enhancement import PromptEnhancementModule
    print("✅ Imported PromptEnhancementModule from modules")
except Exception:
    try:
        sys.path.insert(0, str(_project_root / "modules"))
        from prompt_enhancement import PromptEnhancementModule
        print("✅ Imported PromptEnhancementModule directly")
    except Exception as e:
        print(f"❌ Failed to import PromptEnhancementModule: {e}")
        PromptEnhancementModule = None  # We'll check at runtime and fail nicely if enabled

def masks_postprocess(binary_mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """Remove small connected components"""
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)

def run_yolo_batch(model, image_paths: List[Path], conf: float = 0.25, 
                   iou: float = 0.45, batch_size: int = 32):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = model.predict(batch, conf=conf, iou=iou, verbose=False)
        results.extend(batch_results)
    return results

def load_sam_predictor_compat(checkpoint_path: str, sam_type: str, device: str):
    """Call build_sam_predictor with a compatible signature (auto-detect)."""
    sig = inspect.signature(build_sam_predictor)
    if 'sam_type' in sig.parameters and 'device' in sig.parameters:
        return build_sam_predictor(checkpoint_path, sam_type=sam_type, device=device)
    # Some older signatures may be (checkpoint_path, device='cuda') or (checkpoint_path, )
    if 'device' in sig.parameters:
        return build_sam_predictor(checkpoint_path, device=device)
    # fallback: try single-arg call
    return build_sam_predictor(checkpoint_path)

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 -> BoxRefinement -> PromptEnhancement -> HQ-SAM")
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLOv8 weights")
    parser.add_argument("--ckpt_path", required=True, help="SAM checkpoint path")
    parser.add_argument("--box_refiner_ckpt", default=None, help="Box Refinement checkpoint (optional)")
    parser.add_argument("--images_root", required=True, help="Directory of images")
    parser.add_argument("--out_masks", required=True, help="Output directory for masks")
    parser.add_argument("--sam_type", default="vit_h", help="SAM model type")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, 
                       help="Minimum area ratio for CC filtering")
    parser.add_argument("--batch_size", type=int, default=32, help="YOLO batch size")
    parser.add_argument("--enable_prompt_enhancement", action="store_true", 
                       help="Enable point prompt generation")
    parser.add_argument("--prompt_strategies", nargs='+', 
                       default=['edge_guided', 'texture_contrast'], help="Enabled strategies")
    parser.add_argument("--max_points", type=int, default=8, help="Maximum number of point prompts")
    parser.add_argument("--visualize_prompts", action="store_true", help="Save visualization of point prompts")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")

    images_dir = Path(args.images_root)
    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.JPG"), 
                         *images_dir.glob("*.png"), *images_dir.glob("*.jpeg")])
    if not image_paths:
        print("❌ No images found.", flush=True)
        return
    print(f"✅ Found {len(image_paths)} images.\n")

    # Load SAM predictor (compatibly)
    print(f"Loading SAM predictor from {args.ckpt_path} ...")
    try:
        sam_model_or_predictor = load_sam_predictor_compat(args.ckpt_path, args.sam_type, device)
        print("✅ SAM predictor initialized.\n")
    except Exception as e:
        print(f"❌ Failed to load SAM: {e}", flush=True)
        return

    # Ensure predictor object conforms to SamPredictor API
    SamPredictorClass = None
    try:
        from segment_anything_hq import SamPredictor as HQSamPred
        SamPredictorClass = HQSamPred
    except Exception:
        try:
            from segment_anything import SamPredictor as OrigSamPred
            SamPredictorClass = OrigSamPred
        except Exception:
            SamPredictorClass = None

    if SamPredictorClass is None:
        # assume returned object is already a predictor-like object
        predictor = sam_model_or_predictor
    else:
        if isinstance(sam_model_or_predictor, SamPredictorClass):
            predictor = sam_model_or_predictor
        else:
            # build SamPredictor around model
            predictor = SamPredictorClass(sam_model_or_predictor)

    # Load YOLO
    print(f"Loading YOLO from {args.yolo_weights} ...")
    yolo_model = YOLO(args.yolo_weights)
    if device and device.startswith("cuda"):
        try:
            yolo_model.to(device)
        except Exception:
            pass
    print("✅ YOLO loaded.\n")

    # Load BoxRefiner if available
    box_refiner = None
    if args.box_refiner_ckpt and BoxRefinementModule is not None:
        print(f"Loading Box Refinement from {args.box_refiner_ckpt} ...")
        try:
            box_refiner = BoxRefinementModule(hidden_dim=256)
            state = torch.load(args.box_refiner_ckpt, map_location=device)
            if isinstance(state, dict):
                if "state_dict" in state:
                    sd = state["state_dict"]
                elif "model" in state:
                    sd = state["model"]
                else:
                    sd = state
            else:
                sd = state
            new_sd = {}
            for k, v in sd.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_sd[new_k] = v
            box_refiner.load_state_dict(new_sd, strict=False)
            box_refiner.to(device)
            box_refiner.eval()
            print("✅ Box Refinement loaded.\n")
        except Exception as e:
            print(f"⚠️  Failed to load Box Refinement: {e}", flush=True)
            print("⚠️  Will use original YOLO boxes.\n")
            box_refiner = None
    else:
        print("⚠️  Box Refinement disabled.\n")

    # Initialize Prompt Enhancement Module if enabled
    prompt_enhancer = None
    if args.enable_prompt_enhancement:
        if PromptEnhancementModule is None:
            print("❌ PromptEnhancementModule not found in modules/ but --enable_prompt_enhancement specified.", flush=True)
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
            print(f"✅ Prompt Enhancement initialized with strategies: {args.prompt_strategies}\n")
        except Exception as e:
            print(f"❌ Failed to initialize Prompt Enhancement: {e}\n", flush=True)
            return
    else:
        print("⚠️  Prompt Enhancement disabled.\n")

    vis_dir = None
    if args.visualize_prompts and prompt_enhancer:
        vis_dir = Path(args.out_masks).parent / "visualizations" / "point_prompts"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 Visualizations will be saved to: {vis_dir}\n")

    # Run YOLO detection
    print("Running YOLO detection ...")
    yolo_results = run_yolo_batch(yolo_model, image_paths, conf=args.conf, iou=args.iou, batch_size=args.batch_size)
    out_root = Path(args.out_masks)
    out_root.mkdir(parents=True, exist_ok=True)
    print("✅ YOLO detection complete.\n")

    # Main inference loop
    print("Starting segmentation inference...")
    print(f"{'='*60}\n")

    for img_idx, (img_path, det) in enumerate(zip(image_paths, yolo_results)):
        if (img_idx + 1) % 10 == 0:
            print(f"Processing {img_idx + 1}/{len(image_paths)} ...", flush=True)

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"❌ Failed to read {img_path}", flush=True)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # set image for SAM predictor
        try:
            predictor.set_image(image_rgb)
        except Exception:
            # some variants use predictor.model or other; try to wrap if needed
            try:
                if hasattr(predictor, 'model') and hasattr(predictor.model, 'image_encoder'):
                    # fallback: no set_image method; skip prompt-based methods that need it
                    pass
            except Exception:
                pass

        # get image embedding for box refiner (if applicable)
        image_emb = None
        if box_refiner is not None:
            try:
                image_emb = predictor.get_image_embedding().to(device)
            except Exception:
                image_emb = None

        # Collect YOLO boxes (xyxy)
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
            empty = np.zeros((h, w), dtype=np.uint8)
            save_mask(empty, out_root / f"{img_path.stem}.png")
            continue

        combined = np.zeros((h, w), dtype=np.uint8)

        for box_idx, box in enumerate(boxes_xyxy):
            try:
                box_arr = np.array(box, dtype=np.float32)
                refined_box_abs = box_arr.copy()

                # Box refinement (optional)
                if box_refiner is not None and image_emb is not None:
                    try:
                        bbox_norm = box_arr / np.array([w, h, w, h], dtype=np.float32)
                        bbox_norm_t = torch.tensor(bbox_norm.reshape(1, 4), dtype=torch.float32, device=device)
                        with torch.no_grad():
                            refined_bbox_norm, _ = box_refiner.iterative_refine(
                                image_emb, bbox_norm_t, (h, w), max_iter=3)
                            if refined_bbox_norm is not None:
                                refined_bbox_norm = refined_bbox_norm.squeeze(0).detach().cpu().numpy()
                                refined_box_abs = refined_bbox_norm * np.array([w, h, w, h])
                    except Exception:
                        pass

                # Prompt enhancement (generate point prompts)
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
                    except Exception as e:
                        print(f"⚠️  Prompt generation failed: {e}", flush=True)

                # Prepare inputs for SAM
                refined_boxes_t = torch.tensor([refined_box_abs], dtype=torch.float32, device=device)
                try:
                    transformed = predictor.transform.apply_boxes_torch(refined_boxes_t, (h, w))
                except Exception:
                    # fallback: some predictors expect numpy boxes or different transform API
                    try:
                        transformed = predictor.transform.apply_boxes(refined_box_abs.reshape(1, 4), (h, w))
                    except Exception:
                        transformed = refined_boxes_t

                point_coords_t = None
                point_labels_t = None
                if point_coords_np is not None and len(point_coords_np) > 0:
                    point_coords_t = torch.tensor(point_coords_np, dtype=torch.float32, device=device).unsqueeze(0)
                    point_labels_t = torch.tensor(point_labels_np, dtype=torch.int32, device=device).unsqueeze(0)

                with torch.no_grad():
                    # Some predictors implement predict_torch, others predict; handle both
                    try:
                        masks, scores, _ = predictor.predict_torch(
                            point_coords=point_coords_t,
                            point_labels=point_labels_t,
                            boxes=transformed,
                            multimask_output=False,
                        )
                    except Exception:
                        # fallback to predict (numpy inputs)
                        try:
                            pc = point_coords_np if point_coords_np is not None else None
                            pl = point_labels_np if point_labels_np is not None else None
                            boxes_np = refined_box_abs.reshape(1, 4)
                            masks_all = predictor.predict(
                                point_coords=pc,
                                point_labels=pl,
                                boxes=boxes_np,
                                multimask_output=False,
                            )
                            # predictor.predict may return a list/tuple with masks first
                            if isinstance(masks_all, tuple) or isinstance(masks_all, list):
                                masks = masks_all[0]
                            else:
                                masks = masks_all
                            scores = None
                        except Exception as e:
                            print(f"⚠️  SAM prediction failed fallback: {e}", flush=True)

                            masks = None

                if masks is None or len(masks) == 0:
                    continue

                # masks may be torch tensor or numpy; normalize to uint8 image
                if isinstance(masks, torch.Tensor):
                    m_np = masks[0].squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
                else:
                    # assume numpy array (N, H, W)
                    m_np = (masks[0].astype(np.uint8)) * 255

                combined = np.maximum(combined, m_np)

            except Exception as e:
                print(f"⚠️  Instance failed for {img_path.name}: {e}", flush=True)
                continue

        # Postprocess & save
        combined = masks_postprocess(combined, min_area_ratio=args.min_area_ratio)
        save_mask(combined, out_root / f"{img_path.stem}.png")

    print(f"\n{'='*60}")
    print(f"✅ Inference done. Masks saved to: {out_root}")
    if vis_dir:
        print(f"📊 Visualizations saved to: {vis_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
