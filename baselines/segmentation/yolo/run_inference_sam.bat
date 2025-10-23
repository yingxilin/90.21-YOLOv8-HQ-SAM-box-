@echo off
setlocal

REM YOLO+standard SAM inference (ablation) on Windows
cd /d %~dp0\..\..\..

set YOLO_WEIGHTS=runs\detect\fungi_detection\weights\best.pt
set CKPT_PATH=D:\search\fungi\26\data\models\fungitastic_ckpts\sam_vit_h_4b8939.pth
set IMAGES_ROOT=D:\search\fungi\26\data\FungiTastic-Mini\val\300p
set OUT_MASKS=D:\search\fungi\26\FungiTastic\out\masks_yolo_sam
set SAM_TYPE=vit_h
set CONF=0.35
set IOU=0.6
set MIN_AREA=0.001
set DEVICE=cuda

python baselines\segmentation\yolo\infer_yolo_hqsam.py ^
  --yolo_weights "%YOLO_WEIGHTS%" ^
  --ckpt_path "%CKPT_PATH%" ^
  --images_root "%IMAGES_ROOT%" ^
  --out_masks "%OUT_MASKS%" ^
  --sam_type %SAM_TYPE% ^
  --conf %CONF% ^
  --iou %IOU% ^
  --min_area_ratio %MIN_AREA% ^
  --device %DEVICE% ^
  --save_individual

endlocal

