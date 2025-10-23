#!/usr/bin/env python3
"""
快速测试Box Refinement模块是否正常工作
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_box_refinement():
    """测试Box Refinement模块"""
    print("Testing Box Refinement Module...")
    
    try:
        from modules.box_refinement import BoxRefinementModule, BoxEncoder, OffsetPredictor
        
        # 创建测试数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        B = 2
        H, W = 256, 256
        image_embedding = torch.randn(B, 256, 64, 64).to(device)
        initial_bbox = torch.tensor([[50, 50, 150, 150], [100, 100, 200, 200]], dtype=torch.float32).to(device)
        img_shape = (H, W)
        
        # 测试BoxEncoder
        print("Testing BoxEncoder...")
        box_encoder = BoxEncoder().to(device)
        box_features = box_encoder(initial_bbox, img_shape)
        print(f"BoxEncoder output shape: {box_features.shape}")
        assert box_features.shape == (B, 256), f"Expected (2, 256), got {box_features.shape}"
        print("✓ BoxEncoder test passed")
        
        # 测试OffsetPredictor
        print("Testing OffsetPredictor...")
        offset_predictor = OffsetPredictor().to(device)
        offsets = offset_predictor(box_features)
        print(f"OffsetPredictor output shape: {offsets.shape}")
        assert offsets.shape == (B, 4), f"Expected (2, 4), got {offsets.shape}"
        print("✓ OffsetPredictor test passed")
        
        # 测试完整模块
        print("Testing BoxRefinementModule...")
        model = BoxRefinementModule().to(device)
        
        # 单次前向传播
        offset = model(image_embedding, initial_bbox, img_shape)
        print(f"Single forward output shape: {offset.shape}")
        assert offset.shape == (B, 4), f"Expected (2, 4), got {offset.shape}"
        
        # 迭代精炼
        refined_bbox, history = model.iterative_refine(
            image_embedding, initial_bbox, img_shape,
            max_iter=3, stop_threshold=1.0
        )
        print(f"Refined bbox shape: {refined_bbox.shape}")
        print(f"Number of iterations: {len(history)}")
        assert refined_bbox.shape == (B, 4), f"Expected (2, 4), got {refined_bbox.shape}"
        assert len(history) >= 2, f"Expected at least 2 iterations, got {len(history)}"
        
        print("✓ BoxRefinementModule test passed")
        print("✓ All Box Refinement tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Box Refinement test failed: {e}")
        return False


def test_hqsam_extractor():
    """测试HQ-SAM特征提取器"""
    print("\nTesting HQ-SAM Feature Extractor...")
    
    try:
        from modules.hqsam_feature_extractor import create_hqsam_extractor
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 测试Mock版本
        print("Testing Mock HQ-SAM extractor...")
        mock_extractor = create_hqsam_extractor(
            checkpoint_path="dummy_path",
            model_type="hq_vit_h",
            device=device,
            use_mock=True
        )
        
        features = mock_extractor.extract_features(test_image)
        print(f"Mock extractor output shape: {features.shape}")
        assert features.shape == (1, 256, 64, 64), f"Expected (1, 256, 64, 64), got {features.shape}"
        
        # 测试批量提取
        test_images = [test_image, test_image]
        batch_features = mock_extractor.extract_features_batch(test_images)
        print(f"Batch features length: {len(batch_features)}")
        assert len(batch_features) == 2, f"Expected 2 features, got {len(batch_features)}"
        
        print("✓ HQ-SAM Feature Extractor test passed")
        return True
        
    except Exception as e:
        print(f"✗ HQ-SAM Feature Extractor test failed: {e}")
        return False


def test_training_components():
    """测试训练相关组件"""
    print("\nTesting Training Components...")
    
    try:
        from modules.box_refinement import box_iou_loss
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试IoU损失
        print("Testing IoU loss...")
        pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device)
        target_boxes = torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]], dtype=torch.float32).to(device)
        
        iou_loss = box_iou_loss(pred_boxes, target_boxes)
        print(f"IoU loss: {iou_loss.item():.4f}")
        assert iou_loss.item() >= 0, f"IoU loss should be non-negative, got {iou_loss.item()}"
        
        print("✓ IoU loss test passed")
        
        # 测试L1损失
        print("Testing L1 loss...")
        l1_loss = nn.L1Loss()(pred_boxes, target_boxes)
        print(f"L1 loss: {l1_loss.item():.4f}")
        assert l1_loss.item() >= 0, f"L1 loss should be non-negative, got {l1_loss.item()}"
        
        print("✓ L1 loss test passed")
        print("✓ All training component tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training component test failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Box Refinement Module - Quick Test")
    print("="*60)
    
    all_passed = True
    
    # 测试Box Refinement模块
    if not test_box_refinement():
        all_passed = False
    
    # 测试HQ-SAM特征提取器
    if not test_hqsam_extractor():
        all_passed = False
    
    # 测试训练组件
    if not test_training_components():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Box Refinement module is ready to use.")
        print("\nNext steps:")
        print("1. Update configs/box_refinement_config_local.yaml with your paths")
        print("2. Run: python train_box_refiner.py --config configs/box_refinement_config_local.yaml")
        print("3. Run: python infer_yolo_hqsam_with_refinement.py [your arguments]")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    print("="*60)


if __name__ == "__main__":
    main()