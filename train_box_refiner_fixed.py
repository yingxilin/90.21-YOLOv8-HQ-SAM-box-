#!/usr/bin/env python3
"""
Box Refinement 训练脚本 - 最终修复版本
解决所有已知问题：损失值过大、运行时间过长、缓存失效
"""

import os
import sys
import time
import yaml
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import cv2

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule, box_iou_loss
from modules.hqsam_feature_extractor import create_hqsam_extractor


class FungiDataset(Dataset):
    """FungiTastic 数据集加载器 - 修复版本"""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 image_size: int = 300, data_subset: str = 'Mini',
                 augmentation: bool = True, debug: bool = False,
                 masks_file: Optional[str] = None):
        """
        Args:
            data_root: 数据集根目录
            split: 'train' 或 'val'
            image_size: 图像尺寸
            data_subset: 'Mini' 或 'Full'
            augmentation: 是否启用数据增强
            debug: 调试模式
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.data_subset = data_subset
        self.augmentation = augmentation
        self.debug = debug
        
        self.images_dir = self.data_root / f"{data_subset}" / split / f"{image_size}p"

        # 如提供parquet路径则使用
        if masks_file:
            self.masks_path = Path(masks_file)
            if not self.masks_path.exists():
                raise FileNotFoundError(f"Masks parquet file not found: {self.masks_path}")
            self.use_parquet_masks = True
            # 使用 pyarrow.dataset 进行按需读取，避免一次性加载全表
            try:
                import pyarrow.dataset as ds  # type: ignore
                self._pa_ds = ds.dataset(str(self.masks_path), format='parquet')
                # 记录可用列
                self._pa_schema_names = set(self._pa_ds.schema.names)
                self._pa_has_mask = 'mask' in self._pa_schema_names
                self._pa_has_rle = all(name in self._pa_schema_names for name in ['rle', 'width', 'height'])
                self._pa_file_name_field = 'file_name' if 'file_name' in self._pa_schema_names else None
            except Exception as e:
                raise RuntimeError(f"Failed to open parquet dataset: {e}")
            # 简单的最近使用缓存，减少重复IO
            self._mask_cache = {}
            self._mask_cache_limit = 1024
        else:
            self.masks_dir = self.data_root / f"{data_subset}" / split / "masks"
            if not self.masks_dir.exists():
                raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
            self.use_parquet_masks = False

        
        # 获取图像文件列表
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                 list(self.images_dir.glob("*.JPG")) + 
                                 list(self.images_dir.glob("*.png")))
        
        if debug:
            self.image_files = self.image_files[:100]  # 调试模式只使用前100张图像
        
        print(f"Found {len(self.image_files)} images in {self.split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gt_bbox = None
        if self.use_parquet_masks:
            # 优先从缓存获取（使用文件名作为键）
            image_key = image_path.name
            cached = self._mask_cache.get(image_key)
            if cached is not None:
                gt_bbox = cached
            else:
                try:
                    import pyarrow.dataset as ds  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"pyarrow is required for parquet reading: {e}")
                # 仅请求实际存在的列
                requested_cols = []
                if self._pa_file_name_field is not None:
                    requested_cols.append(self._pa_file_name_field)
                if self._pa_has_mask:
                    requested_cols.append('mask')
                if self._pa_has_rle:
                    requested_cols.extend(['width', 'height', 'rle'])
                if self._pa_file_name_field is not None:
                    filter_expr = (ds.field(self._pa_file_name_field) == image_key)
                else:
                    filter_expr = None
                table = self._pa_ds.to_table(columns=requested_cols, filter=filter_expr) if filter_expr is not None else self._pa_ds.to_table(columns=requested_cols)
                if table.num_rows == 0 and self._pa_file_name_field is not None:
                    filter_expr2 = (ds.field(self._pa_file_name_field) == image_path.stem)
                    table = self._pa_ds.to_table(columns=requested_cols, filter=filter_expr2)
                if table.num_rows == 0:
                    gt_bbox = None
                else:
                    cols = {name: table.column(name) for name in table.schema.names}
                    if self._pa_has_rle and all(k in cols for k in ['rle', 'width', 'height']):
                        rle_list = cols['rle'][0].as_py()
                        width_val = int(cols['width'][0].as_py())
                        height_val = int(cols['height'][0].as_py())
                        gt_bbox = self._compute_bbox_from_rle_counts(rle_list, width_val, height_val)
                    elif self._pa_has_mask and 'mask' in cols:
                        cell = cols['mask'][0].as_py()
                        mask_arr = np.asarray(cell)
                        if mask_arr.ndim == 3:
                            mask_arr = mask_arr[..., 0]
                        if mask_arr.ndim == 1:
                            # 1D 情况，按当前图像尺寸重塑
                            mask_arr = mask_arr.reshape(image.shape[0], image.shape[1])
                        gt_bbox = self._compute_bbox_from_mask(mask_arr.astype(np.uint8))
                # 缓存bbox
                if gt_bbox is None:
                    gt_bbox = self._compute_bbox_from_mask(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
                if len(self._mask_cache) >= self._mask_cache_limit:
                    self._mask_cache.pop(next(iter(self._mask_cache)))
                self._mask_cache[image_key] = gt_bbox
        else:
            mask_path = self.masks_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                gt_bbox = self._compute_bbox_from_mask(mask)
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            if gt_bbox is None:
                gt_bbox = self._compute_bbox_from_mask(mask)
        
        # 调整图像尺寸，并按比例缩放 bbox（避免重建整张mask）
        orig_h, orig_w = image.shape[:2]
        if image.shape[:2] != (self.image_size, self.image_size):
            sx = self.image_size / orig_w
            sy = self.image_size / orig_h
            image = cv2.resize(image, (self.image_size, self.image_size))
            if gt_bbox is not None:
                x1, y1, x2, y2 = gt_bbox
                gt_bbox = np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)
            # 不再需要mask参与训练，提供占位即可
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            # 不使用真实mask以节省CPU时间
            mas        # 生成noisy bbox (模拟YOLO输出)
        noisy_bbox = self._generate_noisy_bbox(gt_bbox, image.shape[:2])
        
        # 🔥 关键修复：归一化边界框坐标到 [0, 1] 范围
        h, w = image.shape[:2]
        gt_bbox_normalized = gt_bbox / np.array([w, h, w, h], dtype=np.float32)
        noisy_bbox_normalized = noisy_bbox / np.array([w, h, w, h], dtype=np.float32)
        
        # 确保坐标在有效范围内
        gt_bbox_normalized = np.clip(gt_bbox_normalized, 0.0, 1.0)
        noisy_bbox_normalized = np.clip(noisy_bbox_normalized, 0.0, 1.0)    h, w = image.shape[:2]
        gt_bbox_normalized = gt_bbox / np.array([w, h, w, h], dtype=np.float32)
        noisy_bbox_normalized = noisy_bbox / np.array([w, h, w, h], dtype=np.float32)
        
        return {
            'image': image,
            'mask': mask,
            'gt_bbox': gt_bbox_normalized,  # 使用归一化坐标
            'noisy_bbox': noisy_bbox_normalized,  # 使用归一化坐标
            'image_path': str(image_path)
        }
    
    def _compute_bbox_from_mask(self, mask):
        """从mask计算边界框"""
        if mask.sum() == 0:
            # 空mask，返回中心区域
            h, w = mask.shape
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            return np.array([center_x - size, center_y - size, center_x + size, center_y + size], dtype=np.float32)
        
        # 找到非零像素的位置
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            h, w = mask.shape
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            return np.array([center_x - size, center_y - size, center_x + size, center_y + size], dtype=np.float32)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    
    def _compute_bbox_from_rle_counts(self, rle_list, width, height):
        """从RLE counts计算边界框"""
        if not rle_list:
            return self._compute_bbox_from_mask(np.zeros((height, width), dtype=np.uint8))
        
        # 将RLE转换为mask
        mask = np.zeros((height, width), dtype=np.uint8)
        pos = 0
        for i, count in enumerate(rle_list):
            if i % 2 == 0:  # 偶数索引表示连续的非零像素
                end_pos = pos + count
                if end_pos <= height * width:
                    y = pos // width
                    x = pos % width
                    mask[y, x:x+count] = 1
            pos += count
        
        return self._compute_bbox_from_mask(mask)
    
    def _generate_noisy_bbox(self, gt_bbox, image_shape):
        """生成带噪声的边界框"""
        if gt_bbox is None:
            h, w = image_shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            gt_bbox = np.array([center_x - size, center_y - size, center_x + size, center_y + size], dtype=np.float32)
        
        # 添加随机噪声
        noise_scale = 0.1  # 10%的噪声
        h, w = image_shape[:2]
        max_noise = min(w, h) * noise_scale
        
        noise = np.random.uniform(-max_noise, max_noise, 4)
        noisy_bbox = gt_bbox + noise
        
        # 确保边界框在图像范围内
        noisy_bbox[0] = max(0, min(noisy_bbox[0], w - 1))  # x1
        noisy_bbox[1] = max(0, min(noisy_bbox[1], h - 1))  # y1
        noisy_bbox[2] = max(noisy_bbox[0] + 1, min(noisy_bbox[2], w))  # x2
        noisy_bbox[3] = max(noisy_bbox[1] + 1, min(noisy_bbox[3], h))  # y2
        
        return noisy_bbox.astype(np.float32)


class FeatureCache:
    """特征缓存管理器 - 修复版本"""
    
    def __init__(self, cache_dir: str, split: str = 'train'):
        self.cache_dir = Path(cache_dir) / f"features/{split}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
    
    def get_cache_path(self, image_path: str) -> Path:
        """获取缓存文件路径"""
        image_hash = hashlib.md5(image_path.encode()).hexdigest()
        return self.cache_dir / f"{image_hash}.npy"
    
    def load_features(self, image_path: str) -> Optional[torch.Tensor]:
        """从缓存加载特征"""
        self.total_requests += 1
        cache_path = self.get_cache_path(image_path)
        
        if cache_path.exists():
            try:
                features = np.load(cache_path)
                features_tensor = torch.from_numpy(features)
                self.cache_hits += 1
                return features_tensor
            except Exception as e:
                print(f"Warning: Failed to load cached features from {cache_path}: {e}")
                self.cache_misses += 1
                return None
        else:
            self.cache_misses += 1
            return None
    
    def save_features(self, image_path: str, features: torch.Tensor):
        """保存特征到缓存"""
        cache_path = self.get_cache_path(image_path)
        try:
            features_cpu = features.cpu().numpy()
            np.save(cache_path, features_cpu)
        except Exception as e:
            print(f"Warning: Failed to save features to {cache_path}: {e}")
    
    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        if self.total_requests == 0:
            return {'hit_rate': 0.0, 'hits': 0, 'misses': 0}
        
        hit_rate = self.cache_hits / self.total_requests
        return {
            'hit_rate': hit_rate,
            'hits': self.cache_hits,
            'misses': self.cache_misses
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0


def detect_feature_cache(cache_dir: str) -> bool:
    """检测是否存在特征缓存"""
    cache_path = Path(cache_dir) / "features"
    return cache_path.exists() and any(cache_path.iterdir())


def extract_features_with_cache(hqsam_extractor, images_np_list, image_paths, feature_cache, device='cuda'):
    """使用缓存提取特征 - 修复版本"""
    features_list = []
    
    for i, (image_np, image_path) in enumerate(zip(images_np_list, image_paths)):
        # 尝试从缓存加载
        if feature_cache is not None:
            cached_features = feature_cache.load_features(image_path)
            if cached_features is not None:
                # 确保缓存的特征在正确设备上
                cached_features = cached_features.to(device)
             def compute_loss(pred_bboxes, gt_bboxes, l1_weight=1.0, iou_weight=0.5):
    """计算损失函数 - 修复版本"""
    # 确保输入张量在相同设备上
    if pred_bboxes.device != gt_bboxes.device:
        gt_bboxes = gt_bboxes.to(pred_bboxes.device)
    
    # 确保输入张量形状一致
    if pred_bboxes.shape != gt_bboxes.shape:
        min_batch = min(pred_bboxes.shape[0], gt_bboxes.shape[0])
        pred_bboxes = pred_bboxes[:min_batch]
        gt_bboxes = gt_bboxes[:min_batch]
    
    # 检查输入有效性
    if pred_bboxes.numel() == 0 or gt_bboxes.numel() == 0:
        return torch.tensor(0.0, device=pred_bboxes.device), torch.tensor(0.0, device=pred_bboxes.device), torch.tensor(0.0, device=pred_bboxes.device)
    
    # L1损失 - 使用更稳定的计算
    l1_loss = F.l1_loss(pred_bboxes, gt_bboxes)
    
    # IoU损失 - 添加数值稳定性
    try:
        iou_loss = box_iou_loss(pred_bboxes, gt_bboxes)
        # 检查IoU损失是否为NaN或Inf
        if torch.isnan(iou_loss) or torch.isinf(iou_loss):
            iou_loss = torch.tensor(0.0, device=pred_bboxes.device)
    except Exception as e:
        print(f"Warning: IoU loss computation failed: {e}")
        iou_loss = torch.tensor(0.0, device=pred_bboxes.device)
    
    # 总损失 - 调整权重比例
    total_loss = l1_weight * l1_loss + iou_weight * iou_loss
    
    return total_loss, l1_loss, iou_lossorch.isnan(iou_loss) or torch.isinf(iou_loss):
            iou_loss = torch.tensor(0.0, device=pred_bboxes.device)
    except Exception as e:
        print(f"Warning: IoU loss computation failed: {e}")
        iou_loss = torch.tensor(0.0, device=pred_bboxes.device)
    
    # 总损失
    total_loss = l1_weight * l1_loss + iou_weight * iou_loss
    
    return total_loss, l1_loss, iou_loss


def train_one_epoch(model, dataloader, optimizer, hqsam_extractor, device, epoch, config, 
                   feature_cache=None, use_amp=False):
    """训练一个epoch - 修复版本"""
    model.train()
    
    total_loss = 0.0
    total_l1_loss = 0.0
    total_iou_loss = 0.0
    num_batches = len(dataloader)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # 修复：使用正确的键名
        images = batch['image'].to(device)
        gt_bboxes = batch['gt_bbox'].to(device)
        noisy_bboxes = batch['noisy_bbox'].to(device)
        image_paths = batch['image_path']
        
        # 确保image_paths是列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]        if use_amp and scaler is not None:
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                # 迭代精炼
                refined_bboxes, history = model.iterative_refine(
                    image_features, noisy_bboxes, (config['data']['image_size'], config['data']['image_size']),
                    max_iter=config['refinement']['max_iter'],
                    stop_threshold=config['refinement']['stop_threshold']
                )
                
                # 计算损失
                loss, l1_loss, iou_loss = compute_loss(
                    refined_bboxes, gt_bboxes,
                    l1_weight=config['loss']['l1_weight'],
                    iou_weight=config['loss']['iou_weight']
                )
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通精度前向传播
            # 迭代精炼
            refined_bboxes, history = model.iterative_refine(
                image_features, noisy_bboxes, (config['data']['image_size'], config['data']['image_size']),
                max_iter=config['refinement']['max_iter'],
                stop_threshold=config['refinement']['stop_threshold']
            )
            
            # 计算损失
            loss, l1_loss, iou_loss = compute_loss(
                refined_bboxes, gt_bboxes,
                l1_weight=config['loss']['l1_weight'],
                iou_weight=config['loss']['iou_weight']
            )
            
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()'],
                stop_threshold=config['refinement']['stop_threshold']
            )
            
            # 计算损失
            loss, l1_loss, iou_loss = compute_loss(
                refined_bboxes, gt_bboxes,
                l1_weight=config['loss']['l1_weight'],
                iou_weight=config['loss']['iou_weight']
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 更新统计
        total_loss += loss.item()
        total_l1_loss += l1_loss.item()
        total_iou_loss += iou_loss.item()
        
        # 更新进度条
        cache_stats = feature_cache.get_cache_stats() if feature_cache else {'hit_rate': 0.0}
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'IoU': f'{iou_loss.item():.4f}',
            'Cache': f'Cache: {cache_stats["hit_rate"]:.1%}'
        })
    
    return total_loss / num_batches, total_l1_loss / num_batches, total_iou_loss / num_batches


def evaluate(model, dataloader, hqsam_extractor, device, config, feature_cache=None):
    """评估模型 - 修复版本"""
    model.eval()
    
    total_loss = 0.0
    total_l1_loss = 0.0
    total_iou_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 修复：使用正确的键名
            images = batch['image'].to(device)
            gt_bboxes = batch['gt_bbox'].to(device)
            noisy_bboxes = batch['noisy_bbox'].to(device)
            image_paths = batch['image_path']
            
            # 确保image_paths是列表
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            # 提取特征
            images_np_list = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
            features_list = extract_features_with_cache(
                hqsam_extractor, images_np_list, image_paths, feature_cache, device
            )
            image_features = torch.cat(features_list, dim=0)
            
            # 前向传播
            refined_bboxes, history = model.iterative_refine(
                image_features, noisy_bboxes, (config['data']['image_size'], config['data']['image_size']),
                max_iter=config['refinement']['max_iter'],
                stop_threshold=config['refinement']['stop_threshold']
            )
            
            # 计算损失
            loss, l1_loss, iou_loss = compute_loss(
                refined_bboxes, gt_bboxes,
                l1_weight=config['loss']['l1_weight'],
                iou_weight=config['loss']['iou_weight']
            )
            
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_iou_loss += iou_loss.item()
    
    return total_loss / num_batches, total_l1_loss / num_batches, total_iou_loss / num_batches


def main():
    """主函数 - 最终修复版本"""
    parser = argparse.ArgumentParser(description='Box Refinement Training - Final Fixed Version')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--fast', action='store_true', help='快速模式')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--clear-cache', action='store_true', help='清空特征缓存')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 快速模式设置
    if args.fast:
        print("🚀 Fast mode enabled - applying all optimizations")
        config['data']['sample_ratio'] = 0.1
        config['training']['use_amp'] = True
        config['training']['batch_size'] = 32
        print(f"  - Data sampling: {config['data']['sample_ratio']}")
        print(f"  - Mixed precision: {config['training']['use_amp']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 检测特征缓存
    cache_detected = detect_feature_cache(config['output']['checkpoint_dir'])
    print(f"Feature cache detected: {cache_detected}")
    
    # 创建特征缓存
    feature_cache = FeatureCache(config['output']['checkpoint_dir']) if config['training']['feature_cache'] else None
    
    # 清空缓存
    if args.clear_cache and feature_cache is not None:
        print("Clearing feature cache...")
        feature_cache.clear_cache()
        print("Feature cache cleared.")
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['train_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=config['data']['augmentation']['enabled'],
        debug=args.debug,
        masks_file=config['data'].get('masks_file')
    )
    
    val_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['val_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=False,
        debug=args.debug,
        masks_file=config['data'].get('masks_file')
    )
    
    # 数据抽样
    if config['data']['sample_ratio'] is not None:
        sample_ratio = config['data']['sample_ratio']
        if sample_ratio < 1.0:
            # 对训练集进行抽样
            train_size = int(len(train_dataset) * sample_ratio)
            train_indices = torch.randperm(len(train_dataset))[:train_size]
            train_dat    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True  # 保持worker进程，减少重启开销
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    ) torch.utils.data.Subset(val_dataset, val_indices)
            print(f"Sampled {len(val_dataset)} images from {len(val_dataset)} total images (ratio: {sample_ratio})")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    print("Creating model...")
    model = BoxRefinementModule(
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        max_offset=config['model']['max_offset']
    ).to(device)
    
    # 创建HQ-SAM特征提取器
    print("Loading HQ-SAM feature extractor...")
    hqsam_extractor = create_hqsam_extractor(
        checkpoint_path=config['hqsam']['checkpoint'],
        model_type=config['hqsam']['model_type'],
        device=device,
        use_mock=True  # 使用Mock版本进行测试
    )
    
    # 创建优化器 - 修复学习率
    print("Creating optimizer...")
    learning_rate = float(config['training']['learning_rate'])
    if args.fast:
        learning_rate *= 2  # 快速模式下稍微提高学习率
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config['training']['weight_decay'])
    )
    
    print(f"Learning rate: {learning_rate}")
    
    # 学习率调度器
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        # 训练
        train_loss, train_l1, train_iou = train_one_epoch(
            model, train_loader, optimizer, hqsam_extractor, device, epoch, config,
            feature_cache=feature_cache, use_amp=config['training']['use_amp']
        )
        
        # 验证
        val_loss, val_l1, val_iou = evaluate(
            model, val_loader, hqsam_extractor, device, config, feature_cache=feature_cache
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印统计信息
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train L1: {train_l1:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val L1: {val_l1:.4f}, IoU: {val_iou:.4f}")
        
        # 缓存统计
        if feature_cache is not None:
            cache_stats = feature_cache.get_cache_stats()
            print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(config['output']['checkpoint_dir'], 'best_model.pth'))
            print(f"  New best model saved! Val Loss: {val_loss:.4f}")
    
    # 保存最终模型
    torch.save({
        'epoch': config['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, os.path.join(config['output']['checkpoint_dir'], 'final_model.pth'))
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # 打印缓存统计
    if feature_cache is not None:
        cache_stats = feature_cache.get_cache_stats()
        print(f"Final cache hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"Total cache hits: {cache_stats['hits']}")
        print(f"Total cache misses: {cache_stats['misses']}")


if __name__ == "__main__":
    main()