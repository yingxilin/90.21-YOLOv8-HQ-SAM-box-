#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低 IoU 样本专用推理脚本（infer_low_iou_only.py）
只对 low_iou_cases.csv 中列出的图像重新执行 YOLO + HQ-SAM + PromptEnhancement 管线。
"""

import argparse
import pandas as pd
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to low_iou_cases.csv")
    parser.add_argument("--images_root", type=str, required=True, help="Root folder containing images (.JPG)")
    parser.add_argument("--script", type=str, required=True, help="Path to the main inference script (infer_with_prompt_enhancement_safe.py)")
    parser.add_argument("--yolo_weights", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--box_refiner_ckpt", type=str, required=True)
    parser.add_argument("--baseline_masks_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_prompt_enhancement", action="store_true")
    parser.add_argument("--visualize_prompts", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    image_names = df["image_name"].tolist()

    src_dir = Path(args.images_root)
    dest_dir = Path(args.out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Total {len(image_names)} low-IoU samples to process.")
    processed = 0

    for name in image_names:
        stem = Path(name).stem
        candidates = list(src_dir.glob(f"{stem}.JPG"))
        if not candidates:
            candidates = list(src_dir.glob(f"{stem}.jpg"))
        if not candidates:
            print(f"⚠️ Image not found: {stem}")
            continue

        img_path = candidates[0]
        print(f"\n🧩 Processing low-IoU image: {img_path.name}")

        cmd = [
            sys.executable,
            args.script,
            "--yolo_weights", args.yolo_weights,
            "--ckpt_path", args.ckpt_path,
            "--box_refiner_ckpt", args.box_refiner_ckpt,
            "--images_root", str(img_path.parent),
            "--out_masks", str(dest_dir),
            "--baseline_masks_dir", args.baseline_masks_dir,
            "--device", args.device
        ]

        if args.enable_prompt_enhancement:
            cmd.append("--enable_prompt_enhancement")
        if args.visualize_prompts:
            cmd.append("--visualize_prompts")

        try:
            subprocess.run(cmd, check=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Error on {img_path.name}: {e}")

    print(f"\n✅ Finished {processed}/{len(image_names)} low-IoU images.")
    print(f"Results saved to: {dest_dir}")

if __name__ == "__main__":
    main()
