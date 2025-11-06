import cv2
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForSegmentation

# ==========================
# 1. 加载 Panoptic DETR
# ==========================
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic", use_safetensors=True)
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic", use_safetensors=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()


def mask_topk_confidence(image_path, topk=1):
    """
    遮挡置信度Top-K的目标区域
    """
    img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
    np_img = np.array(img)
    h, w = np_img.shape[:2]

    # 前向推理
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理成 panoptic segmentation
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)])[0]
    segments_info = result["segments_info"]
    segmentation = result["segmentation"].cpu().numpy().astype(np.uint8)

    if len(segments_info) == 0:
        out_path = os.path.join(output_nomask_dir, "no_masked_" + image_path)
        if not os.path.exists(output_nomask_dir):
            os.makedirs(output_nomask_dir, exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
        print(f"⚠️ 未检测到目标，已保存原图：{out_path}")
        return

    # 排序并选取Top-K segment
    segments_sorted = sorted(segments_info, key=lambda x: x["score"], reverse=True)
    topk_segments = segments_sorted[:topk]
    
    # 构建空mask
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for seg in topk_segments:
        seg_mask = (segmentation == seg["id"]).astype(np.uint8) * 255
        final_mask = cv2.bitwise_or(final_mask, seg_mask)

    # 反转 mask，只保留背景
    final_mask_inv = cv2.bitwise_not(final_mask)
    masked_img = cv2.bitwise_and(np_img, np_img, mask=final_mask_inv)

    out_path = os.path.join(output_masked_dir, f"masked_top{topk}_" + image_path)
    if not os.path.exists(output_masked_dir):
        os.makedirs(output_masked_dir, exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Top-{topk} 置信度目标已遮挡: {out_path}")




output_masked_dir = "/home/yilin/dataset-construct/segmentation/DETR/masked/"
output_nomask_dir = "/home/yilin/dataset-construct/segmentation/DETR/no_masked/"

image_dir = "/data/yilin/train2014/"

input_json = "/home/yilin/dataset-construct/preference_data/filterred_masked_pooler.json"
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    mask_topk_confidence(item["image"], 1)