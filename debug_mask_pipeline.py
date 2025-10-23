#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HQ-SAM + YOLO + BoxRefinement 调试脚本
检查掩码生成异常原因（全黑掩码诊断）
"""

import os, sys, cv2, torch, numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========= 项目路径 =========
SAM_HQ_PATH = r"D:\search\fungi\26\sam-hq"
BOX_REFINER_MODULE = r"D:\search\fungi\26_2\26\modules"
YOLO_WEIGHTS = r"D:\search\fungi\26\FungiTastic\runs\detect\fungi_detection\weights\best.pt"
SAM_CKPT = r"D:\search\fungi\26\data\models\fungitastic_ckpts\sam_hq_vit_h.pth"
BOX_REFINER_CKPT = r"D:\search\fungi\26_2\26\checkpoints\box_refinement\best_model.pth"
TEST_IMAGE = r"D:\search\fungi\26\data\FungiTastic-Mini\val\300p\0-3424495356.JPG"  # 改成任意一张异常图
DEVICE = "cuda"
# ===========================

sys.path.append(SAM_HQ_PATH)
sys.path.append(BOX_REFINER_MODULE)
from segment_anything import sam_model_registry
from box_refinement import BoxRefinementModule


def check_tensor_info(name, t):
    """打印张量统计信息"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    print(f"📊 {name}: shape={t.shape}, mean={t.mean():.5f}, std={t.std():.5f}, min={t.min():.5f}, max={t.max():.5f}")


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 1️⃣ 加载 YOLO
    print("🔹 Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS).to(device).eval()

    # 2️⃣ 加载 HQ-SAM
    print("🔹 Loading HQ-SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(device).eval()

    # 3️⃣ 加载 BoxRefinement
    print("🔹 Loading Box Refinement...")
    box_refiner = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)
    ckpt = torch.load(BOX_REFINER_CKPT, map_location=device)
    box_refiner.load_state_dict(ckpt, strict=False)
    box_refiner.eval()

    # 4️⃣ 读取测试图像
    print(f"\n🖼️ Testing image: {TEST_IMAGE}")
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print("❌ Image not found!")
        return
    H, W = img.shape[:2]
    print(f"Image shape: {H}x{W}")

    # 5️⃣ YOLO 检测
    with torch.no_grad():
        results = yolo.predict(source=str(TEST_IMAGE), conf=0.35, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([])
    print(f"YOLO detections: {len(boxes)} boxes")
    if len(boxes) > 0:
        print(f"First box: {boxes[0]}")

    # 6️⃣ HQ-SAM 编码
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.as_tensor(img_rgb, device=device).permute(2, 0, 1).float() / 255.0
    img_1024 = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False
    )
    with torch.no_grad():
        encoder_out = sam.image_encoder(img_1024)
    image_embedding = encoder_out[0] if isinstance(encoder_out, (tuple, list)) else encoder_out
    interm_emb = encoder_out[1] if isinstance(encoder_out, (tuple, list)) and len(encoder_out) > 1 else None
    check_tensor_info("SAM image_embedding", image_embedding)

        # 7️⃣ Box Refinement
    if len(boxes) == 0:
        print("⚠️ No YOLO boxes found. Skipping refinement.")
        return

    box = boxes[0]
    bbox_norm = torch.tensor([[box[0] / W, box[1] / H, box[2] / W, box[3] / H]], device=device).float()

    # 🔧 确保 image_embedding 是 float32
    if image_embedding.dtype != torch.float32:
        image_embedding = image_embedding.float()

    with torch.no_grad():
        refined_bbox, _ = box_refiner.iterative_refine(image_embedding, bbox_norm, (H, W), max_iter=3)

    refined_bbox = refined_bbox.squeeze(0).cpu().numpy() * np.array([W, H, W, H])
    print(f"Refined box: {refined_bbox}")


    # 8️⃣ HQ-SAM mask decoder
    boxes_1024 = refined_bbox.copy()
    boxes_1024[[0, 2]] *= (1024.0 / W)
    boxes_1024[[1, 3]] *= (1024.0 / H)
    boxes_1024 = torch.tensor([boxes_1024], device=device, dtype=torch.float32)

    with torch.no_grad():
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=boxes_1024, masks=None)
        mask_logits, _ = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_emb,
        )

    check_tensor_info("mask_logits", mask_logits)

    mask = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
    print(f"Mask mean intensity: {mask.mean():.5f}, min={mask.min():.5f}, max={mask.max():.5f}")
    mask_resized = cv2.resize(mask, (W, H))
    mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
    nonzero_ratio = mask_bin.sum() / (H * W * 255)
    print(f"Nonzero pixel ratio: {nonzero_ratio*100:.2f}%")

    cv2.imwrite(str(Path(BOX_REFINER_MODULE) / "debug_mask_output.png"), mask_bin)
    print(f"\n✅ Debug mask saved to: {Path(BOX_REFINER_MODULE) / 'debug_mask_output.png'}")

    # 9️⃣ 最终判断
    print("\n==== DIAGNOSIS ====")
    if len(boxes) == 0:
        print("❌ YOLO 检测无结果 → 检查检测阈值")
    elif image_embedding.abs().mean() < 1e-5:
        print("❌ SAM image_embedding 全零 → 检查 SAM 模型加载或图像预处理")
    elif np.any(refined_bbox < 0) or np.any(refined_bbox > max(H, W)):
        print("⚠️ BoxRefinement 输出越界 → 检查 iterative_refine() 的归一化逻辑")
    elif mask.mean() < 0.001:
        print("❌ mask_decoder 输出全零 → HQ-SAM 没有正确响应提示")
    elif nonzero_ratio < 0.01:
        print("⚠️ 掩码几乎全黑 → 可能提示框太小或无效")
    else:
        print("✅ 一切正常。SAM 输出非空，掩码应可用。")


if __name__ == "__main__":
    main()
