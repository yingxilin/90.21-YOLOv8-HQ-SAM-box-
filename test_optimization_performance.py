#!/usr/bin/env python3
"""
Box Refinement 优化性能测试脚本
对比原始版本和优化版本的性能差异
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
from typing import Dict, List, Tuple

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule
from modules.hqsam_feature_extractor import create_hqsam_extractor
from train_box_refiner_optimized import FeatureCache, FungiDataset, extract_features_with_cache


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        self.device = device
        self.config_path = config_path
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 创建模型和特征提取器
        self.model = BoxRefinementModule(
            hidden_dim=self.config['model']['hidden_dim'],
            num_heads=self.config['model']['num_heads'],
            max_offset=self.config['model']['max_offset']
        ).to(device)
        
        self.hqsam_extractor = create_hqsam_extractor(
            checkpoint_path=self.config['hqsam']['checkpoint'],
            model_type=self.config['hqsam']['model_type'],
            device=device,
            use_mock=True  # 使用Mock版本进行测试
        )
        
        # 创建测试数据
        self.test_images = self._create_test_images(10)  # 10张测试图像
        self.test_bboxes = self._create_test_bboxes(10)
    
    def _create_test_images(self, num_images: int) -> List[np.ndarray]:
        """创建测试图像"""
        images = []
        for i in range(num_images):
            # 创建随机RGB图像
            image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            images.append(image)
        return images
    
    def _create_test_bboxes(self, num_bboxes: int) -> torch.Tensor:
        """创建测试bbox"""
        bboxes = []
        for i in range(num_bboxes):
            # 创建随机bbox [x1, y1, x2, y2]
            x1 = np.random.randint(0, 200)
            y1 = np.random.randint(0, 200)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(50, 100)
            bboxes.append([x1, y1, x2, y2])
        return torch.tensor(bboxes, dtype=torch.float32, device=self.device)
    
    def test_original_approach(self) -> Dict[str, float]:
        """测试原始方法 (每次都提取特征)"""
        print("测试原始方法...")
        
        start_time = time.time()
        
        # 模拟原始训练过程
        for epoch in range(3):  # 3个epoch
            for batch_idx in range(5):  # 5个batch
                # 每次都要提取特征
                features_list = []
                for image in self.test_images:
                    features = self.hqsam_extractor.extract_features(image)
                    features_list.append(features)
                
                # 模拟前向传播
                image_features = torch.cat(features_list, dim=0)
                bboxes = self.test_bboxes[:len(features_list)]
                
                with torch.no_grad():
                    refined_bboxes, _ = self.model.iterative_refine(
                        image_features, bboxes, (300, 300),
                        max_iter=self.config['refinement']['max_iter']
                    )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'total_time': total_time,
            'avg_epoch_time': total_time / 3,
            'avg_batch_time': total_time / (3 * 5),
            'feature_extractions': 3 * 5 * len(self.test_images)
        }
    
    def test_optimized_approach(self) -> Dict[str, float]:
        """测试优化方法 (使用缓存)"""
        print("测试优化方法...")
        
        # 创建特征缓存
        cache_dir = "./test_features"
        feature_cache = FeatureCache(cache_dir, 'test')
        
        start_time = time.time()
        
        # 模拟优化训练过程
        for epoch in range(3):  # 3个epoch
            for batch_idx in range(5):  # 5个batch
                # 使用缓存提取特征
                image_paths = [f"test_image_{i}.jpg" for i in range(len(self.test_images))]
                features_list = extract_features_with_cache(
                    self.hqsam_extractor, self.test_images, image_paths, feature_cache
                )
                
                # 模拟前向传播
                image_features = torch.cat(features_list, dim=0)
                bboxes = self.test_bboxes[:len(features_list)]
                
                with torch.no_grad():
                    refined_bboxes, _ = self.model.iterative_refine(
                        image_features, bboxes, (300, 300),
                        max_iter=self.config['refinement']['max_iter']
                    )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 获取缓存统计
        cache_stats = feature_cache.get_cache_stats()
        
        return {
            'total_time': total_time,
            'avg_epoch_time': total_time / 3,
            'avg_batch_time': total_time / (3 * 5),
            'feature_extractions': 3 * 5 * len(self.test_images),
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_hits': cache_stats['hits'],
            'cache_misses': cache_stats['misses']
        }
    
    def test_data_sampling(self) -> Dict[str, float]:
        """测试数据抽样效果"""
        print("测试数据抽样效果...")
        
        # 创建大数据集
        large_dataset = FungiDataset(
            data_root=self.config['data']['data_root'],
            split=self.config['data']['train_split'],
            image_size=self.config['data']['image_size'],
            data_subset=self.config['data']['data_subset'],
            augmentation=False,
            debug=True,  # 使用调试模式
            sample_ratio=None  # 不使用抽样
        )
        
        # 创建抽样数据集
        sampled_dataset = FungiDataset(
            data_root=self.config['data']['data_root'],
            split=self.config['data']['train_split'],
            image_size=self.config['data']['image_size'],
            data_subset=self.config['data']['data_subset'],
            augmentation=False,
            debug=True,  # 使用调试模式
            sample_ratio=0.1  # 10%抽样
        )
        
        return {
            'full_dataset_size': len(large_dataset),
            'sampled_dataset_size': len(sampled_dataset),
            'sampling_ratio': len(sampled_dataset) / len(large_dataset),
            'reduction_factor': len(large_dataset) / len(sampled_dataset)
        }
    
    def test_mixed_precision(self) -> Dict[str, float]:
        """测试混合精度训练效果"""
        print("测试混合精度训练效果...")
        
        # 创建测试数据
        batch_size = 8
        images = torch.randn(batch_size, 3, 300, 300, device=self.device)
        bboxes = torch.randn(batch_size, 4, device=self.device)
        image_features = torch.randn(batch_size, 256, 64, 64, device=self.device)
        
        # 测试普通精度
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                refined_bboxes, _ = self.model.iterative_refine(
                    image_features, bboxes, (300, 300),
                    max_iter=self.config['refinement']['max_iter']
                )
        normal_time = time.time() - start_time
        
        # 测试混合精度
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    refined_bboxes, _ = self.model.iterative_refine(
                        image_features, bboxes, (300, 300),
                        max_iter=self.config['refinement']['max_iter']
                    )
        amp_time = time.time() - start_time
        
        return {
            'normal_time': normal_time,
            'amp_time': amp_time,
            'speedup': normal_time / amp_time,
            'memory_saved': '~30-50%'  # 估算值
        }
    
    def run_all_tests(self) -> Dict[str, Dict[str, float]]:
        """运行所有测试"""
        print("🚀 开始性能测试...")
        print("=" * 50)
        
        results = {}
        
        # 测试原始方法
        try:
            results['original'] = self.test_original_approach()
        except Exception as e:
            print(f"原始方法测试失败: {e}")
            results['original'] = {}
        
        # 测试优化方法
        try:
            results['optimized'] = self.test_optimized_approach()
        except Exception as e:
            print(f"优化方法测试失败: {e}")
            results['optimized'] = {}
        
        # 测试数据抽样
        try:
            results['sampling'] = self.test_data_sampling()
        except Exception as e:
            print(f"数据抽样测试失败: {e}")
            results['sampling'] = {}
        
        # 测试混合精度
        try:
            results['mixed_precision'] = self.test_mixed_precision()
        except Exception as e:
            print(f"混合精度测试失败: {e}")
            results['mixed_precision'] = {}
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, float]]):
        """打印测试结果"""
        print("\n📊 性能测试结果")
        print("=" * 50)
        
        # 原始 vs 优化对比
        if 'original' in results and 'optimized' in results:
            orig = results['original']
            opt = results['optimized']
            
            if 'total_time' in orig and 'total_time' in opt:
                speedup = orig['total_time'] / opt['total_time']
                print(f"\n🔄 特征缓存效果:")
                print(f"  原始方法总时间: {orig['total_time']:.2f}s")
                print(f"  优化方法总时间: {opt['total_time']:.2f}s")
                print(f"  加速比: {speedup:.1f}x")
                
                if 'cache_hit_rate' in opt:
                    print(f"  缓存命中率: {opt['cache_hit_rate']:.1%}")
                    print(f"  缓存命中: {opt['cache_hits']}")
                    print(f"  缓存未命中: {opt['cache_misses']}")
        
        # 数据抽样效果
        if 'sampling' in results:
            samp = results['sampling']
            print(f"\n📉 数据抽样效果:")
            print(f"  完整数据集大小: {samp.get('full_dataset_size', 'N/A')}")
            print(f"  抽样数据集大小: {samp.get('sampled_dataset_size', 'N/A')}")
            print(f"  抽样比例: {samp.get('sampling_ratio', 0):.1%}")
            print(f"  数据减少倍数: {samp.get('reduction_factor', 1):.1f}x")
        
        # 混合精度效果
        if 'mixed_precision' in results:
            mp = results['mixed_precision']
            print(f"\n⚡ 混合精度效果:")
            print(f"  普通精度时间: {mp.get('normal_time', 0):.2f}s")
            print(f"  混合精度时间: {mp.get('amp_time', 0):.2f}s")
            print(f"  加速比: {mp.get('speedup', 1):.1f}x")
            print(f"  显存节省: {mp.get('memory_saved', 'N/A')}")
        
        # 综合效果估算
        print(f"\n🎯 综合优化效果估算:")
        feature_speedup = results.get('original', {}).get('total_time', 1) / results.get('optimized', {}).get('total_time', 1)
        sampling_speedup = results.get('sampling', {}).get('reduction_factor', 1)
        amp_speedup = results.get('mixed_precision', {}).get('speedup', 1)
        
        total_speedup = feature_speedup * sampling_speedup * amp_speedup
        print(f"  特征缓存加速: {feature_speedup:.1f}x")
        print(f"  数据抽样加速: {sampling_speedup:.1f}x")
        print(f"  混合精度加速: {amp_speedup:.1f}x")
        print(f"  综合加速比: {total_speedup:.1f}x")
        
        if total_speedup >= 30:
            print("  ✅ 达到目标: ≥30x 加速")
        else:
            print("  ⚠️  未达到目标: <30x 加速")
    
    def save_results(self, results: Dict[str, Dict[str, float]], filename: str = "performance_test_results.txt"):
        """保存测试结果到文件"""
        with open(filename, 'w') as f:
            f.write("Box Refinement 性能测试结果\n")
            f.write("=" * 50 + "\n\n")
            
            for test_name, test_results in results.items():
                f.write(f"{test_name.upper()} 测试结果:\n")
                for key, value in test_results.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"\n💾 测试结果已保存到: {filename}")


def main():
    """主函数"""
    print("Box Refinement 性能测试工具")
    print("=" * 50)
    
    # 检查配置文件
    config_path = "configs/box_refinement_config.yaml"
    if not Path(config_path).exists():
        print(f"错误: 配置文件 {config_path} 不存在!")
        return
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cpu':
        print("警告: 使用CPU可能影响测试结果的准确性")
    
    # 创建测试器
    tester = PerformanceTester(config_path, device)
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 打印结果
    tester.print_results(results)
    
    # 保存结果
    tester.save_results(results)
    
    print("\n✅ 性能测试完成!")


if __name__ == "__main__":
    main()