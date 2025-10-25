#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HQ-SAM / SAM 构建器
兼容 FungiTastic 项目 (YOLO + HQ-SAM 推理)
"""

import torch
from pathlib import Path
import sys

# HQ-SAM 路径（根据你项目修改）
SAM_HQ_PATH = r"d:\search\fungi\26\sam-hq"
if SAM_HQ_PATH not in sys.path:
    sys.path.append(SAM_HQ_PATH)

try:
    from segment_anything import sam_model_registry
except Exception as e:
    raise ImportError(
        f"❌ 无法导入 segment_anything，请检查路径是否正确: {SAM_HQ_PATH}"
    ) from e


def build_sam_predictor(ckpt_path: str, model_type: str = "vit_h", device: str = "cuda"):
    """
    构建标准 SAM 预测器
    Args:
        ckpt_path: SAM 权重路径 (.pth)
        model_type: 模型类型 (vit_h / vit_l)
        device: 设备 ('cuda' 或 'cpu')
    Returns:
        predictor: SAM 模型实例
    """
    print(f"🔹 Loading standard SAM model ({model_type}) from {ckpt_path}")
    predictor = sam_model_registry[model_type](checkpoint=ckpt_path)
    predictor.to(device)
    predictor.eval()
    print("✅ SAM predictor ready.")
    return predictor


def build_hqsam_predictor(ckpt_path: str, model_type: str = "hq_vit_h", device: str = "cuda"):
    """
    构建 HQ-SAM 预测器
    Args:
        ckpt_path: HQ-SAM 权重路径 (.pth)
        model_type: 模型类型 ('hq_vit_h' / 'hq_vit_l')
        device: 设备 ('cuda' 或 'cpu')
    Returns:
        predictor: HQ-SAM 模型实例
    """
    print(f"🔹 Loading HQ-SAM model ({model_type}) from {ckpt_path}")
    predictor = sam_model_registry[model_type](checkpoint=ckpt_path)
    predictor.to(device)
    predictor.eval()
    print("✅ HQ-SAM predictor ready.")
    return predictor


# 测试函数（可选）
if __name__ == "__main__":
    model = build_hqsam_predictor(
        ckpt_path=r"D:\search\fungi\26\data\models\fungitastic_ckpts\sam_hq_vit_h.pth",
        model_type="hq_vit_h"
    )
    print("Model loaded:", type(model))
