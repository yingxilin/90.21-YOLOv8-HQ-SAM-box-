#!/usr/bin/env python3
"""
YOLOv8训练脚本：训练蘑菇检测模型

使用ultralytics库训练YOLOv8检测模型，支持多种预训练模型和训练参数。
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def check_gpu_availability():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        print("Warning: GPU not available, training will be slow")
        return False


def load_yolo_config(config_path):
    """加载YOLO配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_yolo_detection(data_yaml, model_name='yolov8l.pt', 
                        imgsz=640, epochs=50, batch=16, 
                        device='cuda', workers=4, **kwargs):
    """
    训练YOLOv8检测模型
    
    Args:
        data_yaml: 数据集配置文件路径
        model_name: 预训练模型名称
        imgsz: 输入图像尺寸
        epochs: 训练轮数
        batch: 批次大小
        device: 设备（cuda/cpu）
        workers: 数据加载器工作进程数
        **kwargs: 其他训练参数
    
    Returns:
        model: 训练好的模型
    """
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # 设置训练参数
    train_args = {
        'data': data_yaml,
        'imgsz': imgsz,
        'epochs': epochs,
        'batch': batch,
        'device': device,
        'workers': workers,
        'project': 'runs/detect',
        'name': 'fungi_detection',
        'exist_ok': True,
        'save': True,
        'save_period': 10,  # 每10个epoch保存一次
        'patience': 20,  # 早停耐心值
        'verbose': True,
    }
    
    # 添加数据增强参数
    augmentation_params = {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'copy_paste': 0.1,
        'mixup': 0.2,
    }
    
    # 合并用户提供的参数
    train_args.update(augmentation_params)
    train_args.update(kwargs)
    
    print("Training parameters:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    print("Starting training...")
    results = model.train(**train_args)
    
    print("Training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Last model saved at: {model.trainer.last}")
    
    return model


def validate_model(model, data_yaml, split='val'):
    """
    验证训练好的模型
    
    Args:
        model: 训练好的模型
        data_yaml: 数据集配置文件
        split: 验证集分割
    """
    print(f"Validating model on {split} set...")
    
    # 加载验证数据
    val_results = model.val(data=data_yaml, split=split)
    
    print("Validation results:")
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall: {val_results.box.mr:.4f}")
    
    return val_results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detection model")
    parser.add_argument("--data_yaml", required=True, help="Path to dataset YAML file")
    parser.add_argument("--model", default="yolov8l.pt", 
                       choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                       help="Pretrained model to use")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    
    # 数据增强参数
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV hue augmentation")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV saturation augmentation")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV value augmentation")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation augmentation")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--copy_paste", type=float, default=0.1, help="Copy-paste augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup augmentation probability")
    
    args = parser.parse_args()
    
    # 检查GPU可用性
    if args.device == 'cuda':
        check_gpu_availability()
    
    # 检查数据配置文件
    if not os.path.exists(args.data_yaml):
        print(f"Error: Data YAML file not found: {args.data_yaml}")
        sys.exit(1)
    
    # 加载数据集配置
    data_config = load_yolo_config(args.data_yaml)
    print(f"Dataset config: {data_config}")
    
    # 准备训练参数
    train_kwargs = {
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'translate': args.translate,
        'scale': args.scale,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'copy_paste': args.copy_paste,
        'mixup': args.mixup,
    }
    
    if args.resume:
        train_kwargs['resume'] = args.resume
    
    # 开始训练
    model = train_yolo_detection(
        data_yaml=args.data_yaml,
        model_name=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        **train_kwargs
    )
    
    # 验证模型
    if args.validate:
        validate_model(model, args.data_yaml)
    
    print("Training script completed!")


if __name__ == "__main__":
    main()