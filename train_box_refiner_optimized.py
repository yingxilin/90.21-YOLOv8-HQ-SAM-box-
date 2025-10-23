#!/usr/bin/env python3
"""
Box Refinement 训练脚本 - 稳定增强版（带最优模型保存 + 可视化曲线）
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt  # ✅ 新增：可视化用

# 添加 modules 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from modules.box_refinement import BoxRefinementModule, box_iou_loss
from modules.hqsam_feature_extractor import create_hqsam_extractor


# ======================================================
# 数据集
# ======================================================
class FungiDataset(Dataset):
    """FungiTastic 数据集加载器（稳定版）"""

    def __init__(self, data_root: str, split: str = 'train', image_size: int = 300,
                 data_subset: str = 'Mini', augmentation: bool = True,
                 debug: bool = False):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.data_subset = data_subset
        self.augmentation = augmentation
        self.debug = debug
        self.images_dir = self.data_root / f"{data_subset}" / split / f"{image_size}p"

        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) +
                                  list(self.images_dir.glob("*.png")) +
                                  list(self.images_dir.glob("*.JPG")))

        if debug:
            self.image_files = self.image_files[:100]
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        print(f"Found {len(self.image_files)} images in {self.split} split")

    def __len__(self):
        return len(self.image_files)

    def _compute_bbox_from_mask(self, mask):
        if mask.sum() == 0:
            h, w = mask.shape
            cx, cy = w // 2, h // 2
            size = min(w, h) // 4
            return np.array([cx - size, cy - size, cx + size, cy + size], dtype=np.float32)
        coords = np.where(mask > 0)
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _generate_noisy_bbox(self, gt_bbox, image_shape):
        h, w = image_shape[:2]
        noise_scale = 0.1
        max_noise = min(w, h) * noise_scale
        noise = np.random.uniform(-max_noise, max_noise, 4)
        noisy_bbox = gt_bbox + noise
        noisy_bbox[0] = np.clip(noisy_bbox[0], 0, w - 1)
        noisy_bbox[1] = np.clip(noisy_bbox[1], 0, h - 1)
        noisy_bbox[2] = np.clip(noisy_bbox[2], noisy_bbox[0] + 1, w)
        noisy_bbox[3] = np.clip(noisy_bbox[3], noisy_bbox[1] + 1, h)
        return noisy_bbox.astype(np.float32)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        gt_bbox = self._compute_bbox_from_mask(mask)
        noisy_bbox = self._generate_noisy_bbox(gt_bbox, image.shape)

        h, w = image.shape[:2]
        gt_bbox /= np.array([w, h, w, h], dtype=np.float32)
        noisy_bbox /= np.array([w, h, w, h], dtype=np.float32)

        return {
            'image': torch.tensor(image.astype(np.float32).transpose(2, 0, 1)),
            'gt_bbox': torch.tensor(gt_bbox.astype(np.float32)),
            'noisy_bbox': torch.tensor(noisy_bbox.astype(np.float32)),
            'image_path': str(image_path)
        }


# ======================================================
# 损失函数
# ======================================================
def compute_loss(pred_bboxes, gt_bboxes, l1_weight=1.0, iou_weight=0.5):
    l1_loss = F.l1_loss(pred_bboxes, gt_bboxes)
    iou_loss = box_iou_loss(pred_bboxes, gt_bboxes)
    total_loss = l1_weight * l1_loss + iou_weight * iou_loss
    return total_loss, l1_loss, iou_loss


# ======================================================
# 训练与验证
# ======================================================
def train_one_epoch(model, dataloader, optimizer, hqsam_extractor, device, epoch, config, use_amp=False):
    model.train()
    total_loss = total_l1 = total_iou = 0.0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        images = batch['image'].clone().detach().to(device)
        gt_bboxes = batch['gt_bbox'].clone().detach().to(device)
        noisy_bboxes = batch['noisy_bbox'].clone().detach().to(device)

        features_list = [hqsam_extractor.extract_features(img.unsqueeze(0)) for img in images]
        image_features = torch.cat(features_list, dim=0)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            refined_bboxes, _ = model.iterative_refine(
                image_features, noisy_bboxes,
                (config['data']['image_size'], config['data']['image_size']),
                max_iter=config['refinement']['max_iter'],
                stop_threshold=config['refinement']['stop_threshold']
            )
            loss, l1_loss, iou_loss = compute_loss(
                refined_bboxes, gt_bboxes,
                config['loss']['l1_weight'], config['loss']['iou_weight']
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_iou += iou_loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    n = len(dataloader)
    return total_loss / n, total_l1 / n, total_iou / n


def evaluate(model, dataloader, hqsam_extractor, device, config):
    model.eval()
    total_loss = total_l1 = total_iou = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].clone().detach().to(device)
            gt_bboxes = batch['gt_bbox'].clone().detach().to(device)
            noisy_bboxes = batch['noisy_bbox'].clone().detach().to(device)
            features_list = [hqsam_extractor.extract_features(img.unsqueeze(0)) for img in images]
            image_features = torch.cat(features_list, dim=0)
            refined_bboxes, _ = model.iterative_refine(
                image_features, noisy_bboxes,
                (config['data']['image_size'], config['data']['image_size']),
                max_iter=config['refinement']['max_iter'],
                stop_threshold=config['refinement']['stop_threshold']
            )
            loss, l1_loss, iou_loss = compute_loss(refined_bboxes, gt_bboxes)
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_iou += iou_loss.item()
    n = len(dataloader)
    return total_loss / n, total_l1 / n, total_iou / n


# ======================================================
# 主函数（增强）
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = FungiDataset(config['data']['data_root'], config['data']['train_split'],
                                 config['data']['image_size'], config['data']['data_subset'])
    val_dataset = FungiDataset(config['data']['data_root'], config['data']['val_split'],
                               config['data']['image_size'], config['data']['data_subset'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = BoxRefinementModule(
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        max_offset=config['model']['max_offset']
    ).to(device)

    hqsam_extractor = create_hqsam_extractor(
        checkpoint_path=config['hqsam']['checkpoint'],
        model_type=config['hqsam']['model_type'],
        device=device,
        use_mock=True
    )

    lr = float(config['training']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ✅ 新增：最优模型保存路径与记录变量
    save_dir = Path(r"D:\search\fungi\26_2\26\checkpoints\box_refinement")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"
    best_val_loss = float("inf")

    train_losses, val_losses = [], []

    print(f"\n🚀 Training Start (fast={args.fast}, debug={args.debug})")
    print(f" - LR: {lr}\n - Epochs: {config['training']['epochs']}\n - Batch size: {config['training']['batch_size']}")

    for epoch in range(config['training']['epochs']):
        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, hqsam_extractor, device, epoch, config)
        val_loss, _, _ = evaluate(model, val_loader, hqsam_extractor, device, config)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")

        # ✅ 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved (val_loss={val_loss:.6f}) → {best_model_path}")

    # ✅ 训练完成后绘制曲线
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png")
    print(f"\n📈 Loss curve saved to {save_dir / 'loss_curve.png'}")
    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
