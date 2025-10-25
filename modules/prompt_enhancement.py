#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PromptEnhancementModule (Adaptive version)
针对 FungiTastic 项目中 "低 IoU" 样本优化
策略: 
  - 对简单样本：使用中心正点 + 边缘负点
  - 对复杂样本（颜色混杂、小目标、低对比）：使用 KMeans 聚类颜色提取正点，并在边缘/高对比处取负点
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
import os


class PromptEnhancementModule:
    def __init__(self, margin_ratio: float = 0.05, visualize_dir: str = None):
        """
        Args:
            margin_ratio: 负点相对 bbox 边缘的距离比例
            visualize_dir: 若提供路径，则保存提示点可视化图像
        """
        self.margin_ratio = margin_ratio
        self.visualize_dir = visualize_dir
        if visualize_dir:
            os.makedirs(visualize_dir, exist_ok=True)
        print(f"[PromptEnhancement] Adaptive version initialized (margin={margin_ratio})")

    def _compute_color_complexity(self, crop):
        """计算颜色复杂度 (标准差均值)"""
        if crop is None or crop.size == 0:
            return 0
        stds = np.std(crop.reshape(-1, 3), axis=0)
        return float(np.mean(stds))

    def _generate_adaptive_prompts(self, image, bbox):
        """核心增强逻辑：颜色聚类 + 边缘检测"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return None, None

        # Step 1: KMeans聚类找代表色中心
        pixels = crop.reshape(-1, 3)
        k = 3 if len(pixels) > 500 else 2
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42).fit(pixels)
        centers = np.uint8(kmeans.cluster_centers_)
        labels = kmeans.labels_

        # 找颜色最“深”的簇作为可能的菌体（假设菌体更暗）
        cluster_brightness = centers.mean(axis=1)
        dark_cluster = np.argmin(cluster_brightness)
        mask_dark = (labels == dark_cluster).reshape(crop.shape[:2]).astype(np.uint8)

        # 提取该簇的中心点作为正点
        ys, xs = np.where(mask_dark > 0)
        if len(xs) == 0:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            cx, cy = xs.mean() + x1, ys.mean() + y1

        # Step 2: 负点（边缘与亮区）
        edge = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 80, 160)
        edges_y, edges_x = np.where(edge > 0)
        neg_points = []
        if len(edges_x) > 0:
            idx = np.random.choice(len(edges_x), min(4, len(edges_x)), replace=False)
            for i in idx:
                neg_points.append([x1 + edges_x[i], y1 + edges_y[i]])

        # Step 3: 边缘负点
        m = self.margin_ratio
        neg_points += [
            [x1 + (x2 - x1) * m, y1 + (y2 - y1) * m],
            [x2 - (x2 - x1) * m, y1 + (y2 - y1) * m],
            [x1 + (x2 - x1) * m, y2 - (y2 - y1) * m],
            [x2 - (x2 - x1) * m, y2 - (y2 - y1) * m],
        ]

        pos_points = [[cx, cy]]
        pos_labels = [1]
        neg_labels = [0] * len(neg_points)
        pts = np.array(pos_points + neg_points, dtype=np.float32)
        lbls = np.array(pos_labels + neg_labels, dtype=np.int32)

        return pts, lbls

    def generate_point_prompts(self, image, bbox, case_name=None):
        """主入口：根据 bbox 生成点提示"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / (h * w)
        crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        color_complexity = self._compute_color_complexity(crop)

        # 判断是否复杂样本
        use_adaptive = (area_ratio < 0.05) or (color_complexity > 30)

        if use_adaptive:
            pts, lbls = self._generate_adaptive_prompts(image, bbox)
        else:
            # 简单样本：中心 + 边缘负点
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            m = self.margin_ratio
            neg_points = [
                [x1 + (x2 - x1) * m, y1 + (y2 - y1) * m],
                [x2 - (x2 - x1) * m, y1 + (y2 - y1) * m],
                [x1 + (x2 - x1) * m, y2 - (y2 - y1) * m],
                [x2 - (x2 - x1) * m, y2 - (y2 - y1) * m],
            ]
            pts = np.array([[cx, cy]] + neg_points, dtype=np.float32)
            lbls = np.array([1] + [0] * len(neg_points), dtype=np.int32)

        # 可视化保存
        if self.visualize_dir and case_name:
            vis = image.copy()
            for (x, y), l in zip(pts, lbls):
                color = (0, 255, 0) if l == 1 else (0, 0, 255)
                cv2.circle(vis, (int(x), int(y)), 4, color, -1)
            cv2.imwrite(os.path.join(self.visualize_dir, f"{case_name}_prompts.jpg"), vis)

        return {"point_coords": pts, "point_labels": lbls}
