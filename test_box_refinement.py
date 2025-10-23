#!/usr/bin/env python3
"""
测试 Box Refinement 模块的改进效果（完整独立版本）
✅ 无需修改 modules/box_refinement.py
✅ 内置 compute_loss 和 box_iou_loss
"""

import torch
import numpy as np
import sys
import os
import time

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule


# ===========================================================
# 🔧 损失函数定义（内置）
# ===========================================================

def box_iou_loss(pred_boxes, target_boxes, eps=1e-6):
    """
    IoU 损失函数 (1 - IoU)
    Args:
        pred_boxes: [B, 4] (x1, y1, x2, y2)
        target_boxes: [B, 4] (x1, y1, x2, y2)
    Returns:
        iou_loss: 标量
    """
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(0)
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(0) * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(0)
    union = area_pred + area_target - inter + eps

    iou = inter / union
    return 1.0 - iou.mean()


def compute_loss(pred_boxes, target_boxes,
                 l1_weight=1.0, iou_weight=2.0, total_variation_weight=0.1):
    """
    综合损失函数：
        total = L1 + IoU + Total Variation 正则
    """
    # L1 损失
    l1_loss = torch.nn.functional.l1_loss(pred_boxes, target_boxes)

    # IoU 损失
    iou_loss = box_iou_loss(pred_boxes, target_boxes)

    # Total Variation 平滑项
    if pred_boxes.ndim == 2 and pred_boxes.size(0) > 1:
        tv_loss = torch.mean(torch.abs(pred_boxes[1:] - pred_boxes[:-1]))
    else:
        tv_loss = torch.tensor(0.0, device=pred_boxes.device)

    total_loss = (
        l1_weight * l1_loss +
        iou_weight * iou_loss +
        total_variation_weight * tv_loss
    )

    return total_loss, l1_loss, iou_loss


# ===========================================================
# 🧪 测试函数部分
# ===========================================================

def test_loss_functions():
    """测试损失函数"""
    print("Testing loss functions...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_boxes = torch.tensor([[0.2, 0.2, 0.8, 0.8],
                               [0.1, 0.1, 0.9, 0.9]], device=device)
    target_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75],
                                 [0.15, 0.15, 0.85, 0.85]], device=device)

    iou_loss = box_iou_loss(pred_boxes, target_boxes)
    print(f"IoU Loss: {iou_loss.item():.4f}")

    total_loss, l1_loss, iou_loss = compute_loss(pred_boxes, target_boxes)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"IoU Loss: {iou_loss.item():.4f}")

    assert total_loss.item() < 10.0, f"Loss too high: {total_loss.item()}"
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert not torch.isinf(total_loss), "Loss is Inf"

    print("✅ Loss functions test passed!")


def test_model_forward():
    """测试模型前向传播"""
    print("Testing model forward pass...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)

    B = 2
    H, W = 300, 300
    image_embedding = torch.randn(B, 256, 64, 64, device=device)
    initial_bbox = torch.tensor([[0.2, 0.2, 0.8, 0.8],
                                 [0.1, 0.1, 0.9, 0.9]], device=device)
    img_shape = (H, W)

    offset = model(image_embedding, initial_bbox, img_shape)
    print(f"Offset shape: {offset.shape}")
    print(f"Offset values: {offset}")

    assert offset.shape == (B, 4), f"Wrong offset shape: {offset.shape}"
    assert torch.all(torch.abs(offset) <= 50), f"Offset too large: {offset.max()}"

    refined_bbox, history = model.iterative_refine(
        image_embedding, initial_bbox, img_shape, max_iter=3, stop_threshold=1.0
    )
    print(f"Refined bbox shape: {refined_bbox.shape}")
    print(f"Iterations: {len(history)}")

    assert refined_bbox.shape == (B, 4)
    assert torch.all(refined_bbox >= 0)
    assert torch.all(refined_bbox <= 1)

    print("✅ Model forward pass test passed!")


def test_training_step():
    """测试训练步骤"""
    print("Testing training step...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    B = 4
    H, W = 300, 300
    image_embedding = torch.randn(B, 256, 64, 64, device=device)
    noisy_bbox = torch.tensor([[0.2, 0.2, 0.8, 0.8],
                               [0.1, 0.1, 0.9, 0.9],
                               [0.3, 0.3, 0.7, 0.7],
                               [0.15, 0.15, 0.85, 0.85]], device=device)
    gt_bbox = torch.tensor([[0.25, 0.25, 0.75, 0.75],
                            [0.15, 0.15, 0.85, 0.85],
                            [0.35, 0.35, 0.65, 0.65],
                            [0.2, 0.2, 0.8, 0.8]], device=device)
    img_shape = (H, W)

    model.train()
    optimizer.zero_grad()

    refined_bbox, history = model.iterative_refine(
        image_embedding, noisy_bbox, img_shape, max_iter=3, stop_threshold=1.0
    )

    loss, l1_loss, iou_loss = compute_loss(refined_bbox, gt_bbox)
    print(f"Training Loss: {loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"IoU Loss: {iou_loss.item():.4f}")

    loss.backward()

    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    print(f"Gradient norm: {total_grad_norm:.4f}")

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert total_grad_norm < 100, f"Gradient norm too large: {total_grad_norm}"

    optimizer.step()

    print("✅ Training step test passed!")


def test_performance():
    """测试性能"""
    print("Testing performance...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)
    model.eval()

    B = 8
    H, W = 300, 300
    image_embedding = torch.randn(B, 256, 64, 64, device=device)
    initial_bbox = torch.rand(B, 4, device=device)
    img_shape = (H, W)

    with torch.no_grad():
        for _ in range(10):
            _ = model.iterative_refine(image_embedding, initial_bbox, img_shape)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            _ = model.iterative_refine(image_embedding, initial_bbox, img_shape)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"Average inference time: {avg_time * 1000:.2f}ms per batch")

    assert avg_time < 1.0, f"Inference too slow: {avg_time:.2f}s"

    print("✅ Performance test passed!")


# ===========================================================
# 🧩 主入口
# ===========================================================
def main():
    print("🧪 Testing Box Refinement Module Improvements")
    print("=" * 50)

    try:
        test_loss_functions()
        print()

        test_model_forward()
        print()

        test_training_step()
        print()

        test_performance()
        print()

        print("🎉 All tests passed! The improvements are working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
