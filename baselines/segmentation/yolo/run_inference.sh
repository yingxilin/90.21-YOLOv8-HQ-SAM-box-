#!/bin/bash
# YOLO+HQ-SAM推理运行脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径
YOLO_WEIGHTS="runs/detect/fungi_detection/weights/best.pt"
SAM_CKPT_PATH="D:/search/fungi/24/FungiTastic/ckpts"

# 数据路径
IMAGES_ROOT="D:/search/fungi/24/FungiTastic/FungiTastic-Mini/test/300p"
OUT_MASKS="D:/search/fungi/24/data/FungiTastic/masks_yolo_hqsam"

# 推理参数
SAM_TYPE="hq_vit_h"  # 可选: hq_vit_h, hq_vit_l, vit_h, vit_l
CONF=0.35
IOU=0.6
MIN_AREA_RATIO=0.001
DEVICE="cuda"

echo "Starting YOLO+HQ-SAM inference..."
echo "YOLO weights: $YOLO_WEIGHTS"
echo "SAM checkpoint: $SAM_CKPT_PATH"
echo "Images root: $IMAGES_ROOT"
echo "Output masks: $OUT_MASKS"
echo "SAM type: $SAM_TYPE"

# 检查YOLO权重是否存在
if [ ! -f "$YOLO_WEIGHTS" ]; then
    echo "Error: YOLO weights not found: $YOLO_WEIGHTS"
    echo "Please train the YOLO model first using run_training.sh"
    exit 1
fi

# 运行推理
python baselines/segmentation/yolo/infer_yolo_hqsam.py \
    --yolo_weights "$YOLO_WEIGHTS" \
    --ckpt_path "$SAM_CKPT_PATH" \
    --images_root "$IMAGES_ROOT" \
    --out_masks "$OUT_MASKS" \
    --sam_type "$SAM_TYPE" \
    --conf $CONF \
    --iou $IOU \
    --min_area_ratio $MIN_AREA_RATIO \
    --device "$DEVICE" \
    --save_individual

echo "Inference completed!"
echo "Results saved to: $OUT_MASKS"