#!/usr/bin/env python3
"""
YOLO标注生成脚本：将二值掩码转换为YOLO检测框标签

从FungiTastic数据集的二值掩码生成YOLO格式的检测标签：
- 输入：原始图像和二值掩码（parquet格式）
- 处理：连通域分析 → 外接矩形 → YOLO格式
- 输出：YOLO格式标签文件和数据集配置
"""

import os
import sys
import argparse
import numpy as np
import ast
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
from skimage.measure import label, regionprops
import shutil
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class MaskTable:
    """Cache a single parquet table in memory and provide fast lookup by image id."""
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        # Detect columns once
        possible_id_cols = ['image_id', 'image_name', 'filename', 'file_name', 'name', 'id']
        possible_mask_cols = ['mask', 'segmentation', 'masks', 'mask_data', 'rle']
        self.id_col = None
        for col in possible_id_cols:
            if col in self.df.columns:
                self.id_col = col
                break
        if self.id_col is None:
            self.id_col = self.df.columns[0]
        self.mask_col = None
        for col in possible_mask_cols:
            if col in self.df.columns:
                self.mask_col = col
                break
        if self.mask_col is None:
            # Fallback: find array-like column
            for col in self.df.columns:
                if col != self.id_col and not self.df[col].empty:
                    sample_data = self.df[col].iloc[0]
                    if isinstance(sample_data, (list, np.ndarray)):
                        self.mask_col = col
                        break
        if self.mask_col is None:
            raise ValueError(f"No mask column found in {parquet_path}")

        # Normalize RLE column to list type if present
        if self.mask_col == 'rle' and isinstance(self.df['rle'].iloc[0], str):
            self.df['rle'] = self.df['rle'].apply(ast.literal_eval)

        # If RLE-based, we may have multiple rows per image; aggregate
        if self.mask_col == 'rle':
            # Ensure height/width columns exist
            height_col = 'height' if 'height' in self.df.columns else None
            width_col = 'width' if 'width' in self.df.columns else None
            agg = {
                'rle': list,
            }
            if height_col:
                agg[height_col] = 'first'
            if width_col:
                agg[width_col] = 'first'
            self.df = self.df.groupby(self.id_col).agg(agg).reset_index()
            # Remember names
            self._height_col = height_col
            self._width_col = width_col
        else:
            self._height_col = None
            self._width_col = None

        # Build fast lookup by normalized basename (case-insensitive)
        def _norm_basename(s: str) -> str:
            try:
                base = Path(s).name
            except Exception:
                base = s
            return str(base).lower()

        self.df['__norm_basename__'] = self.df[self.id_col].astype(str).apply(_norm_basename)
        self.df['__basename_no_ext__'] = self.df['__norm_basename__'].apply(lambda x: x.rsplit('.', 1)[0])
        # Index dictionaries
        self._by_basename = {row['__norm_basename__']: i for i, row in self.df.iterrows()}
        self._by_stem = {row['__basename_no_ext__']: i for i, row in self.df.iterrows()}

    def get_mask(self, image_name: str):
        # Normalize query
        q_basename = Path(image_name).name.lower()
        q_stem = q_basename.rsplit('.', 1)[0]

        # Direct match on normalized basename
        idx = self._by_basename.get(q_basename)
        if idx is None:
            # Fallback: match on stem (no extension)
            idx = self._by_stem.get(q_stem)
        if idx is None:
            # Last resort: try exact id column equality (legacy)
            row = self.df[self.df[self.id_col] == image_name]
            if row.empty:
                row = self.df[self.df[self.id_col] == q_basename]
            if row.empty:
                row = self.df[self.df[self.id_col] == q_stem]
            if row.empty:
                return None
        if idx is not None:
            row = self.df.iloc[[idx]]

        mask_data = row[self.mask_col].iloc[0]
        if self.mask_col == 'rle':
            # Build merged binary mask from list of RLE runs
            rles = mask_data
            if not isinstance(rles, list) or len(rles) == 0:
                return None
            if self._height_col is None or self._width_col is None:
                return None
            height = int(row[self._height_col].iloc[0])
            width = int(row[self._width_col].iloc[0])
            # Decode CVAT-style RLEs and merge
            binary = np.zeros((height, width), dtype=bool)
            for rle in rles:
                if hasattr(rle, 'tolist'):
                    rle = rle.tolist()
                # last 4 numbers are bbox; strip them
                counts = rle[:-4] if len(rle) >= 4 else rle
                flat = np.zeros(height * width, dtype=np.uint8)
                current_position = 0
                current_value = 0
                for c in counts:
                    flat[current_position:current_position + c] = current_value
                    current_position += c
                    current_value = 1 - current_value
                mask = flat.reshape((height, width)).astype(bool)
                binary |= mask
            return binary.astype(np.uint8)
        else:
            if isinstance(mask_data, np.ndarray):
                return mask_data
            if isinstance(mask_data, list):
                return np.array(mask_data)
            return None


def load_mask_from_parquet(parquet_path, image_name):
    """
    从parquet文件中加载指定图像的掩码
    
    Args:
        parquet_path: parquet文件路径
        image_name: 图像文件名（不含扩展名）
    
    Returns:
        mask: 二值掩码数组 (H, W)
    """
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Parquet columns: {df.columns.tolist()}")
        print(f"Looking for image: {image_name}")
        
        # 尝试不同的列名组合
        possible_id_cols = ['image_id', 'image_name', 'filename', 'name', 'id']
        possible_mask_cols = ['mask', 'segmentation', 'masks', 'mask_data']
        
        # 查找图像ID列
        id_col = None
        for col in possible_id_cols:
            if col in df.columns:
                id_col = col
                break
        
        if id_col is None:
            # 如果没找到明确的ID列，使用第一列
            id_col = df.columns[0]
            print(f"Using first column as ID: {id_col}")
        
        # 查找掩码列
        mask_col = None
        for col in possible_mask_cols:
            if col in df.columns:
                mask_col = col
                break
        
        if mask_col is None:
            # 如果没找到明确的掩码列，查找包含数组数据的列
            for col in df.columns:
                if col != id_col:
                    sample_data = df[col].iloc[0]
                    if isinstance(sample_data, (list, np.ndarray)):
                        mask_col = col
                        break
        
        if mask_col is None:
            print(f"Error: No mask column found in {parquet_path}")
            return None
        
        print(f"Using ID column: {id_col}, Mask column: {mask_col}")
        
        # 查找匹配的图像
        row = df[df[id_col] == image_name]
        if row.empty:
            # 尝试去掉文件扩展名
            image_name_no_ext = image_name.split('.')[0]
            row = df[df[id_col] == image_name_no_ext]
        
        if row.empty:
            print(f"Warning: No mask found for {image_name} in {parquet_path}")
            return None
        
        mask_data = row[mask_col].iloc[0]
        if isinstance(mask_data, np.ndarray):
            return mask_data
        elif isinstance(mask_data, list):
            return np.array(mask_data)
        else:
            print(f"Warning: Unexpected mask data type for {image_name}: {type(mask_data)}")
            return None
        
    except Exception as e:
        print(f"Error loading mask for {image_name}: {e}")
        return None


def mask_to_yolo_boxes(mask, min_area_ratio=0.001):
    """
    将二值掩码转换为YOLO格式的边界框
    
    Args:
        mask: 二值掩码 (H, W)
        min_area_ratio: 最小面积比例阈值
    
    Returns:
        boxes: YOLO格式边界框列表 [(class, x_center, y_center, width, height), ...]
    """
    if mask is None:
        return []
    
    # 确保掩码是二值的
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # 连通域分析
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    boxes = []
    h, w = mask.shape
    min_area = min_area_ratio * h * w
    
    for region in regions:
        # 过滤小区域
        if region.area < min_area:
            continue
            
        # 获取边界框
        minr, minc, maxr, maxc = region.bbox
        
        # 转换为YOLO格式 (归一化坐标)
        x_center = (minc + maxc) / 2.0 / w
        y_center = (minr + maxr) / 2.0 / h
        width = (maxc - minc) / w
        height = (maxr - minr) / h
        
        # 确保坐标在[0,1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        # 只保留有效的框（面积大于0）
        if width > 0 and height > 0:
            boxes.append((0, x_center, y_center, width, height))  # class=0 for mushroom
    
    return boxes


def _draw_boxes_on_image(image_bgr: np.ndarray, boxes, color=(0, 255, 0)):
    for (_, xc, yc, w, h) in boxes:
        H, W = image_bgr.shape[:2]
        x_center = xc * W
        y_center = yc * H
        bw = w * W
        bh = h * H
        x1 = int(round(x_center - bw / 2))
        y1 = int(round(y_center - bh / 2))
        x2 = int(round(x_center + bw / 2))
        y2 = int(round(y_center + bh / 2))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    return image_bgr


def process_split(images_root, masks_file, out_images_dir, out_labels_dir, 
                 split, size, min_area_ratio=0.001, limit=None, sample_every=1,
                 viz_out_dir: Path = None, viz_max: int = 0, num_workers: int = 0,
                 only_labeled: bool = True):
    """
    处理单个数据集分割（train/val）
    
    Args:
        images_root: 原始图像根目录
        masks_file: 掩码parquet文件路径
        out_images_dir: 输出图像目录
        out_labels_dir: 输出标签目录
        split: 分割名称（train/val）
        size: 图像尺寸（如300p）
        min_area_ratio: 最小面积比例
    """
    split_images_dir = Path(images_root) / split / size
    
    if not split_images_dir.exists():
        print(f"Warning: Image directory {split_images_dir} does not exist")
        return
    
    if not os.path.exists(masks_file):
        print(f"Warning: Mask file {masks_file} does not exist")
        return
    
    # 创建输出目录
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(split_images_dir.glob("*.jpg")) + list(split_images_dir.glob("*.JPG")) + list(split_images_dir.glob("*.png"))
    
    print(f"Processing {split} split with {len(image_files)} images...")
    print(f"Using mask file: {masks_file}")
    if viz_out_dir is not None:
        print(f"Visualization dir: {viz_out_dir}, viz_max={viz_max}")
    
    # Cache parquet once
    try:
        mt = MaskTable(masks_file)
        print(f"Mask table columns: {mt.df.columns.tolist()}")
        print(f"Using id_col='{mt.id_col}', mask_col='{mt.mask_col}'")
    except Exception as e:
        print(f"Error loading mask parquet: {e}")
        return

    # Optionally filter to only images that have labels in parquet
    if only_labeled:
        labeled_basenames = set(mt._by_basename.keys())
        labeled_stems = set(mt._by_stem.keys())
        before = len(image_files)
        kept = []
        for p in image_files:
            bn = p.name.lower()
            st = bn.rsplit('.', 1)[0]
            if (bn in labeled_basenames) or (st in labeled_stems):
                kept.append(p)
        image_files = kept
        after = len(image_files)
        print(f"Filtered to labeled images: {after}/{before} remain")

    # Sampling (after filtering)
    if sample_every > 1:
        image_files = image_files[::sample_every]
    if limit is not None and limit > 0:
        image_files = image_files[:limit]

    processed_count = 0
    skipped_count = 0
    viz_saved = 0

    def process_one(img_path: Path):
        nonlocal viz_saved
        img_name = img_path.stem
        # Try multiple keys: full basename, full path relative to subset/split/size
        mask = mt.get_mask(img_path.name)
        if mask is None:
            # Attempt with relative path pattern commonly used in parquet (e.g., 'train/300p/filename.JPG')
            rel_key = f"{split}/{size}/{img_path.name}"
            mask = mt.get_mask(rel_key)
        if mask is None:
            # Attempt lowercase jpg vs JPG differences by replacing extension
            name_lower_jpg = img_path.stem + '.jpg'
            mask = mt.get_mask(name_lower_jpg)
        if mask is None:
            name_upper_jpg = img_path.stem + '.JPG'
            mask = mt.get_mask(name_upper_jpg)
        if mask is None:
            return (img_path, None, "no_mask")
        boxes = mask_to_yolo_boxes(mask, min_area_ratio)
        # copy image
        out_img_path = out_images_dir / img_path.name
        shutil.copy2(img_path, out_img_path)
        # write labels
        label_path = out_labels_dir / f"{img_name}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                class_id, x_center, y_center, width, height = box
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        # optional visualization
        if viz_out_dir is not None and viz_max > 0 and viz_saved < viz_max:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is not None:
                vis = _draw_boxes_on_image(image_bgr.copy(), boxes)
                viz_out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(viz_out_dir / img_path.name), vis)
                viz_saved += 1
        return (img_path, True, "ok")

    if num_workers and num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(process_one, p): p for p in image_files}
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}"):
                pass
        # recount by scanning outputs
        processed_count = len([p for p in image_files if (out_labels_dir / f"{p.stem}.txt").exists()])
        skipped_count = len(image_files) - processed_count
    else:
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            _, ok, reason = process_one(img_path)
            if ok:
                processed_count += 1
            else:
                skipped_count += 1

    print(f"{split}: Processed {processed_count} images, skipped {skipped_count} images")


def create_yolo_config(out_root, train_dir, val_dir):
    """
    创建YOLO数据集配置文件
    
    Args:
        out_root: 输出根目录
        train_dir: 训练图像目录
        val_dir: 验证图像目录
    """
    config = {
        'path': str(out_root),
        'train': train_dir,
        'val': val_dir,
        'names': ['mushroom']
    }
    
    config_path = Path(out_root) / "yolo_fungi.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created YOLO config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO labels from binary masks")
    parser.add_argument("--images_root", required=True, help="Root directory of images")
    parser.add_argument("--masks_file", required=True, help="Path to mask parquet file")
    parser.add_argument("--out_root", required=True, help="Output root directory")
    parser.add_argument("--sizes", default="300p", help="Image sizes to process (comma-separated)")
    parser.add_argument("--splits", default="train,val", help="Splits to process (comma-separated)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, 
                       help="Minimum area ratio for filtering small regions")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images (per split)")
    parser.add_argument("--sample_every", type=int, default=1, help="Sample every k images (per split)")
    parser.add_argument("--num_workers", type=int, default=0, help="Thread workers for parallel IO/label writing")
    parser.add_argument("--viz_out", default="", help="Optional directory to save visualization images")
    parser.add_argument("--viz_max", type=int, default=16, help="Max number of visualizations to save per split")
    
    args = parser.parse_args()
    
    # 解析参数
    sizes = [s.strip() for s in args.sizes.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Images root: {args.images_root}")
    print(f"Masks file: {args.masks_file}")
    print(f"Output root: {args.out_root}")
    print(f"Sizes: {sizes}")
    print(f"Splits: {splits}")
    print(f"Min area ratio: {args.min_area_ratio}")
    
    # 可选可视化输出目录
    viz_out_dir = Path(args.viz_out) if args.viz_out else None

    # 处理每个尺寸和分割
    for size in sizes:
        for split in splits:
            out_images_dir = out_root / "images" / split
            out_labels_dir = out_root / "labels" / split
            
            process_split(
                args.images_root, 
                args.masks_file,
                out_images_dir,
                out_labels_dir,
                split,
                size,
                args.min_area_ratio,
                limit=(args.limit if args.limit > 0 else None),
                sample_every=max(1, args.sample_every),
                viz_out_dir=viz_out_dir,
                viz_max=args.viz_max,
                num_workers=max(0, args.num_workers)
            )
    
    # 创建YOLO配置文件
    create_yolo_config(
        out_root,
        "images/train",
        "images/val"
    )
    
    print("YOLO label generation completed!")


if __name__ == "__main__":
    main()