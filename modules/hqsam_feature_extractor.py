"""
HQ-SAM 特征提取器封装
只提取图像特征,不进行分割
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple
import sys
import os

# 添加segmentation目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'segmentation'))

try:
    from hqsam.build_hqsam import build_sam_predictor
except ImportError:
    print("Warning: Could not import build_sam_predictor. Make sure HQ-SAM is properly installed.")
    build_sam_predictor = None


class HQSAMFeatureExtractor:
    """HQ-SAM 特征提取器，只用于提取图像特征"""
    
    def __init__(self, checkpoint_path: str, model_type: str = 'vit_h', device: str = 'cuda'):
        """
        Args:
            checkpoint_path: HQ-SAM 预训练权重路径
            model_type: 'vit_h', 'vit_l', 或 'vit_b'
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model_type = model_type
        
        if build_sam_predictor is None:
            raise ImportError("HQ-SAM not available. Please install HQ-SAM first.")
        
        # 加载 HQ-SAM 模型
        print(f"Loading HQ-SAM model ({model_type}) from {checkpoint_path}...")
        self.predictor = build_sam_predictor(checkpoint_path, sam_type=model_type, device=device)
        
        # 冻结所有参数
        self._freeze_parameters()
        
        print("HQ-SAM feature extractor initialized successfully.")
    
    def _freeze_parameters(self):
        """冻结所有参数，只用于特征提取"""
        for param in self.predictor.model.parameters():
            param.requires_grad = False
        self.predictor.model.eval()
    
    @torch.no_grad()
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            image: (H, W, 3) - numpy array, RGB, [0, 255]
        Returns:
            features: (1, 256, 64, 64) - Tensor
        """
        # 设置图像
        self.predictor.set_image(image)
        
        # 获取图像特征
        # HQ-SAM 的 image encoder 输出特征图
        features = self.predictor.features  # (1, 256, 64, 64)
        
        return features
    
    @torch.no_grad()
    def extract_features_batch(self, images: list) -> list:
        """
        批量提取图像特征
        
        Args:
            images: List[np.ndarray] - 图像列表
        Returns:
            features_list: List[torch.Tensor] - 特征列表
        """
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)
        return features_list
    
    def preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: (H, W, 3) - 输入图像
            target_size: (H, W) - 目标尺寸，如果为None则保持原尺寸
        Returns:
            processed_image: (H, W, 3) - 预处理后的图像
        """
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 已经是RGB格式
            processed_image = image.copy()
        else:
            # 转换为RGB
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸（如果需要）
        if target_size is not None:
            processed_image = cv2.resize(processed_image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 确保数据类型为uint8
        processed_image = processed_image.astype(np.uint8)
        
        return processed_image
    
    def get_feature_shape(self) -> Tuple[int, int, int, int]:
        """
        获取特征图的形状
        
        Returns:
            shape: (B, C, H, W) - 特征图形状
        """
        return (1, 256, 64, 64)  # HQ-SAM ViT-H 的特征图形状


class MockHQSAMFeatureExtractor:
    """Mock HQ-SAM 特征提取器，用于测试"""
    
    def __init__(self, checkpoint_path: str, model_type: str = 'vit_h', device: str = 'cuda'):
        self.device = device
        self.model_type = model_type
        print(f"Mock HQ-SAM feature extractor initialized (model_type: {model_type})")
    
    @torch.no_grad()
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """返回随机特征用于测试"""
        # 返回随机特征 (1, 256, 64, 64)
        features = torch.randn(1, 256, 64, 64, device=self.device)
        return features
    
    @torch.no_grad()
    def extract_features_batch(self, images: list) -> list:
        """批量提取随机特征"""
        features_list = []
        for _ in images:
            features = torch.randn(1, 256, 64, 64, device=self.device)
            features_list.append(features)
        return features_list
    
    def preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """预处理图像（Mock版本）"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image.copy()
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def get_feature_shape(self) -> Tuple[int, int, int, int]:
        return (1, 256, 64, 64)


def create_hqsam_extractor(checkpoint_path: str, model_type: str = 'vit_h', 
                          device: str = 'cuda', use_mock: bool = False) -> HQSAMFeatureExtractor:
    """
    创建HQ-SAM特征提取器
    
    Args:
        checkpoint_path: 检查点路径
        model_type: 模型类型
        device: 设备
        use_mock: 是否使用Mock版本（用于测试）
    
    Returns:
        HQSAMFeatureExtractor 实例
    """
    if use_mock or build_sam_predictor is None:
        return MockHQSAMFeatureExtractor(checkpoint_path, model_type, device)
    else:
        return HQSAMFeatureExtractor(checkpoint_path, model_type, device)


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 测试Mock版本
    print("Testing Mock HQ-SAM feature extractor...")
    mock_extractor = MockHQSAMFeatureExtractor("dummy_path", device=device)
    
    features = mock_extractor.extract_features(test_image)
    print(f"Feature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")
    print(f"Feature device: {features.device}")
    
    # 测试批量提取
    test_images = [test_image, test_image]
    batch_features = mock_extractor.extract_features_batch(test_images)
    print(f"Batch features length: {len(batch_features)}")
    
    print("Mock test passed!")
    
    # 测试真实版本（如果可用）
    try:
        print("\nTesting real HQ-SAM feature extractor...")
        real_extractor = HQSAMFeatureExtractor("dummy_path", device=device)
        print("Real HQ-SAM test would run here if checkpoint was available.")
    except Exception as e:
        print(f"Real HQ-SAM test skipped: {e}")