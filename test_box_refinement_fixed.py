#!/usr/bin/env python3
"""
测试修复后的 Box Refinement 模块
"""

import torch
import numpy as np
import sys
import os

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule, box_iou_loss

def test_box_refinement():
    """测试Box Refinement模块"""
    print("=" * 60)
    print("Testing Box Refinement Module (Fixed Version)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 测试参数
    B = 4
    H, W = 300, 300
    
    # 1. 测试模型初始化
    print("1. Testing model initialization...")
    try:
        model = BoxRefinementModule(
            hidden_dim=256,
            num_heads=8,
            max_offset=0.1  # 归一化坐标
        ).to(device)
        print("   ✓ Model initialized successfully")
        print(f"   - Hidden dim: 256")
        print(f"   - Num heads: 8")
        print(f"   - Max offset: 0.1 (normalized)\n")
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}\n")
        return False
    
    # 2. 测试数据准备
    print("2. Testing data preparation...")
    try:
        # 创建测试数据（归一化坐标）
        image_embedding = torch.randn(B, 256, 64, 64).to(device)
        # 归一化坐标 [0, 1]
        initial_bbox = torch.tensor([
            [0.2, 0.2, 0.6, 0.6],
            [0.3, 0.3, 0.7, 0.7],
            [0.1, 0.1, 0.5, 0.5],
            [0.4, 0.4, 0.8, 0.8]
        ], dtype=torch.float32).to(device)
        
        gt_bbox = torch.tensor([
            [0.25, 0.25, 0.65, 0.65],
            [0.32, 0.32, 0.72, 0.72],
            [0.15, 0.15, 0.55, 0.55],
            [0.42, 0.42, 0.82, 0.82]
        ], dtype=torch.float32).to(device)
        
        img_shape = (H, W)
        
        print("   ✓ Test data prepared")
        print(f"   - Batch size: {B}")
        print(f"   - Image shape: {H}x{W}")
        print(f"   - Image embedding shape: {image_embedding.shape}")
        print(f"   - Bbox shape: {initial_bbox.shape}")
        print(f"   - Bbox range: [{initial_bbox.min().item():.3f}, {initial_bbox.max().item():.3f}] (normalized)\n")
    except Exception as e:
        print(f"   ✗ Data preparation failed: {e}\n")
        return False
    
    # 3. 测试单次前向传播
    print("3. Testing single forward pass...")
    try:
        with torch.no_grad():
            offset = model(image_embedding, initial_bbox, img_shape)
        
        print("   ✓ Forward pass successful")
        print(f"   - Offset shape: {offset.shape}")
        print(f"   - Offset range: [{offset.min().item():.6f}, {offset.max().item():.6f}]")
        print(f"   - Mean absolute offset: {offset.abs().mean().item():.6f}\n")
        
        # 检查offset范围是否合理
        if offset.abs().max().item() > 0.2:
            print(f"   ⚠ Warning: Offset seems too large (max: {offset.abs().max().item():.3f})\n")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}\n")
        return False
    
    # 4. 测试迭代精炼
    print("4. Testing iterative refinement...")
    try:
        with torch.no_grad():
            refined_bbox, history = model.iterative_refine(
                image_embedding, 
                initial_bbox, 
                img_shape,
                max_iter=3,
                stop_threshold=0.01
            )
        
        print("   ✓ Iterative refinement successful")
        print(f"   - Refined bbox shape: {refined_bbox.shape}")
        print(f"   - Number of iterations: {len(history)}")
        print(f"   - Bbox range: [{refined_bbox.min().item():.3f}, {refined_bbox.max().item():.3f}]")
        
        # 打印每次迭代的变化
        print("\n   Iteration history:")
        for i in range(len(history)):
            if i == 0:
                print(f"     Iter {i} (initial): mean={history[i].mean(0)}")
            else:
                change = (history[i] - history[i-1]).abs().mean().item()
                print(f"     Iter {i}: mean={history[i].mean(0)}, change={change:.6f}")
        print()
        
    except Exception as e:
        print(f"   ✗ Iterative refinement failed: {e}\n")
        return False
    
    # 5. 测试损失函数
    print("5. Testing loss function...")
    try:
        # 测试IoU损失
        iou_loss = box_iou_loss(refined_bbox, gt_bbox)
        
        print("   ✓ Loss computation successful")
        print(f"   - IoU loss: {iou_loss.item():.4f}")
        
        # 检查损失范围
        if iou_loss.item() < 0 or iou_loss.item() > 1:
            print(f"   ✗ Error: IoU loss out of range [0, 1]: {iou_loss.item():.4f}\n")
            return False
        
        # 计算L1损失
        l1_loss = torch.nn.functional.l1_loss(refined_bbox, gt_bbox)
        print(f"   - L1 loss: {l1_loss.item():.4f}")
        
        # 检查L1损失范围（归一化坐标下应该较小）
        if l1_loss.item() > 0.5:
            print(f"   ⚠ Warning: L1 loss seems high: {l1_loss.item():.4f}\n")
        else:
            print(f"   ✓ L1 loss is within expected range\n")
        
    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}\n")
        return False
    
    # 6. 测试梯度反向传播
    print("6. Testing gradient backpropagation...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # 前向传播
        refined_bbox, _ = model.iterative_refine(
            image_embedding, 
            initial_bbox, 
            img_shape
        )
        
        # 计算损失
        loss = box_iou_loss(refined_bbox, gt_bbox)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("   ✓ Gradient backpropagation successful")
        print(f"   - Loss: {loss.item():.4f}")
        
        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    print(f"   - {name}: grad_norm={grad_norm:.6f}")
        
        if not has_grad:
            print("   ⚠ Warning: No gradients found\n")
        else:
            print()
        
    except Exception as e:
        print(f"   ✗ Gradient backpropagation failed: {e}\n")
        return False
    
    # 7. 测试坐标系一致性
    print("7. Testing coordinate system consistency...")
    try:
        # 检查所有bbox是否在[0, 1]范围内
        all_bboxes = torch.cat([initial_bbox, refined_bbox, gt_bbox], dim=0)
        
        if all_bboxes.min() < 0 or all_bboxes.max() > 1:
            print(f"   ✗ Error: Bboxes out of normalized range [0, 1]")
            print(f"   - Min: {all_bboxes.min().item():.3f}")
            print(f"   - Max: {all_bboxes.max().item():.3f}\n")
            return False
        
        # 检查x2 > x1, y2 > y1
        valid_boxes = (
            (refined_bbox[:, 2] > refined_bbox[:, 0]) & 
            (refined_bbox[:, 3] > refined_bbox[:, 1])
        ).all()
        
        if not valid_boxes:
            print("   ✗ Error: Invalid box format (x2 <= x1 or y2 <= y1)\n")
            return False
        
        print("   ✓ Coordinate system is consistent")
        print("   - All bboxes in [0, 1] range")
        print("   - All boxes have valid format (x2 > x1, y2 > y1)\n")
        
    except Exception as e:
        print(f"   ✗ Coordinate system check failed: {e}\n")
        return False
    
    print("=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_box_refinement()
    exit(0 if success else 1)
