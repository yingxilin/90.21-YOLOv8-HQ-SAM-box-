@echo off
setlocal

REM YOLOv8 training on Windows
cd /d %~dp0\..\..\..

set DATA_YAML=datasets\yolo_fungi\yolo_fungi.yaml
set MODEL=yolov8l.pt
set IMGSZ=640
set EPOCHS=50
set BATCH=16
set DEVICE=cuda

python baselines\segmentation\yolo\train_yolo.py ^
  --data_yaml "%DATA_YAML%" ^
  --model "%MODEL%" ^
  --imgsz %IMGSZ% ^
  --epochs %EPOCHS% ^
  --batch %BATCH% ^
  --device %DEVICE% ^
  --validate

endlocal

