#!/bin/bash
# YOLOv8训练运行脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 数据集路径
DATA_YAML="datasets/yolo_fungi/yolo_fungi.yaml"

# 检查数据集配置是否存在
if [ ! -f "$DATA_YAML" ]; then
    echo "Error: Dataset YAML file not found: $DATA_YAML"
    echo "Please run make_yolo_labels.py first to generate the dataset"
    exit 1
fi

# 训练参数
MODEL="yolov8l.pt"  # 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
IMGSZ=640
EPOCHS=50
BATCH=16
DEVICE="cuda"
WORKERS=4

# 数据增强参数
HSV_H=0.015
HSV_S=0.7
HSV_V=0.4
TRANSLATE=0.1
SCALE=0.5
FLIPLR=0.5
MOSAIC=1.0
COPY_PASTE=0.1
MIXUP=0.2

echo "Starting YOLOv8 training..."
echo "Dataset: $DATA_YAML"
echo "Model: $MODEL"
echo "Image size: $IMGSZ"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH"
echo "Device: $DEVICE"

# 运行训练
python baselines/segmentation/yolo/train_yolo.py \
    --data_yaml "$DATA_YAML" \
    --model "$MODEL" \
    --imgsz $IMGSZ \
    --epochs $EPOCHS \
    --batch $BATCH \
    --device "$DEVICE" \
    --workers $WORKERS \
    --validate \
    --hsv_h $HSV_H \
    --hsv_s $HSV_S \
    --hsv_v $HSV_V \
    --translate $TRANSLATE \
    --scale $SCALE \
    --fliplr $FLIPLR \
    --mosaic $MOSAIC \
    --copy_paste $COPY_PASTE \
    --mixup $MIXUP

echo "Training completed!"
echo "Check runs/detect/fungi_detection/ for results"