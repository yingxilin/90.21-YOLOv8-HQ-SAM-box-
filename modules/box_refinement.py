"""
Iterative Box Refinement Module
论文参考: RoBox-SAM (2024)
创新点: 利用 HQ-SAM 图像特征自动学习 YOLO bbox 的偏移量
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, d_model = x.shape
        position = torch.arange(d_model, device=x.device).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float()
            * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(B, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position[0::2] * div_term)
        pe[:, 1::2] = torch.cos(position[0::2] * div_term)
        return x + pe


class BoxEncoder(nn.Module):
    """将 bbox 坐标编码为高维特征向量 (输入为像素或归一化坐标均可)"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = 8
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, bboxes, img_shape):
        B = bboxes.shape[0]
        H, W = img_shape

        # 将 bbox 归一化到 [0,1] 范围
        x1 = torch.clamp(bboxes[:, 0] / W, 0.0, 1.0)
        y1 = torch.clamp(bboxes[:, 1] / H, 0.0, 1.0)
        x2 = torch.clamp(bboxes[:, 2] / W, 0.0, 1.0)
        y2 = torch.clamp(bboxes[:, 3] / H, 0.0, 1.0)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)

        box_features = torch.stack([x1, y1, x2, y2, cx, cy, w, h], dim=1)
        encoded = self.mlp(box_features)
        encoded = self.pos_encoding(encoded)
        return encoded


class OffsetPredictor(nn.Module):
    """预测 bbox 的偏移量（输出归一化单位，范围 [-max_offset, max_offset]）"""

    def __init__(self, hidden_dim=256, max_offset=0.1):
        super().__init__()
        self.max_offset = max_offset
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, features):
        raw = self.mlp(features)             # (B,4)
        offsets = torch.tanh(raw) * self.max_offset
        return offsets


class BoxRefinementModule(nn.Module):
    """完整的 Box Refinement 模块（统一归一化坐标系）"""

    def __init__(self, hidden_dim=256, num_heads=8, max_offset=0.1):
        super().__init__()
        self.box_encoder = BoxEncoder(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.offset_predictor = OffsetPredictor(hidden_dim, max_offset)
        self.hidden_dim = hidden_dim
        self.max_offset = max_offset
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                if getattr(module, "in_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if getattr(module, "in_proj_bias", None) is not None:
                    nn.init.constant_(module.in_proj_bias, 0)

    def forward(self, image_embedding, bbox, img_shape):
        """
        单步前向推理：
        输入 bbox 可以是像素坐标或归一化坐标，
        内部始终转换到归一化后进行偏移预测。
        """
        B = bbox.shape[0]
        H, W = img_shape

        # 转换为归一化坐标
        x1 = bbox[:, 0] / W
        y1 = bbox[:, 1] / H
        x2 = bbox[:, 2] / W
        y2 = bbox[:, 3] / H
        bbox_norm = torch.stack([x1, y1, x2, y2], dim=1)

        box_features = self.box_encoder(bbox, img_shape)
        image_features = image_embedding.view(B, self.hidden_dim, -1).transpose(1, 2)
        box_features = box_features.unsqueeze(1)
        refined_features, _ = self.cross_attention(
            query=box_features, key=image_features, value=image_features
        )
        refined_features = refined_features.squeeze(1)
        offset_norm = self.offset_predictor(refined_features)  # (B,4)
        return offset_norm

    def iterative_refine(
        self, image_embedding, initial_bbox, img_shape, max_iter=3, stop_threshold=1e-3
    ):
        """
        迭代式边界框精炼 (内部使用归一化坐标)
        """
        B = initial_bbox.shape[0]
        H, W = img_shape

        # 判断输入坐标类型
        is_pixel = float(initial_bbox.max()) > 1.01
        if is_pixel:
            bbox = initial_bbox.clone()
            bbox[:, [0, 2]] /= W
            bbox[:, [1, 3]] /= H
        else:
            bbox = initial_bbox.clone()

        current_bbox = bbox
        history = [current_bbox.clone().detach()]

        for _ in range(max_iter):
            offset = self.forward(image_embedding, current_bbox * torch.tensor([W, H, W, H], device=current_bbox.device), (H, W))
            current_bbox = current_bbox + offset

            # clamp 到 [0,1]
            current_bbox = torch.clamp(current_bbox, 0.0, 1.0)

            # 保证宽高正值
            eps = 1.0 / max(H, W)
            x1, y1, x2, y2 = current_bbox[:, 0], current_bbox[:, 1], current_bbox[:, 2], current_bbox[:, 3]
            x2 = torch.maximum(x2, x1 + eps)
            y2 = torch.maximum(y2, y1 + eps)
            current_bbox = torch.stack([x1, y1, x2, y2], dim=1)

            history.append(current_bbox.clone().detach())

            offset_magnitude = torch.norm(offset, dim=1).mean()
            if offset_magnitude.item() < stop_threshold:
                break

        # 返回结果
        if is_pixel:
            final_bbox = current_bbox * torch.tensor([W, H, W, H], device=current_bbox.device)
        else:
            final_bbox = current_bbox

        final_bbox = final_bbox + 0.0 * offset  # 保持梯度连接
        return final_bbox, history


def box_iou_loss(pred_boxes, target_boxes):
    """计算 bbox IoU loss (1 - IoU)"""
    if pred_boxes.device != target_boxes.device:
        target_boxes = target_boxes.to(pred_boxes.device)
    if pred_boxes.shape != target_boxes.shape:
        n = min(pred_boxes.shape[0], target_boxes.shape[0])
        pred_boxes = pred_boxes[:n]
        target_boxes = target_boxes[:n]

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    )
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    )
    union = pred_area + target_area - intersection
    iou = intersection / (union + 1e-7)
    return 1.0 - iou.mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    H, W = 256, 256
    image_embedding = torch.randn(B, 256, 64, 64).to(device)
    initial_bbox = torch.tensor(
        [[50, 50, 150, 150], [100, 100, 200, 200]], dtype=torch.float32
    ).to(device)
    model = BoxRefinementModule().to(device)

    offset = model(image_embedding, initial_bbox, (H, W))
    print(f"Offset shape: {offset.shape}")
    refined_bbox, history = model.iterative_refine(image_embedding, initial_bbox, (H, W))
    print(f"Refined bbox shape: {refined_bbox.shape}")
    print(f"Number of iterations: {len(history)}")
    print("✅ Test passed!")
