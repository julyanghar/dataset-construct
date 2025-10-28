import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForSegmentation

# ==========================
# ✅ 配置部分
# ==========================
MODEL_NAME = "facebook/detr-resnet-50-panoptic"
IMAGE_DIR = "/home/yilin/dataset/train2014/"
INPUT_JSON = "/home/yilin/dataset-construct/preference_data/pref_data.json"

TOP_K = 2  # ← 你可以在这里调节 K 值

OUTPUT_MASKED_DIR = f"/home/yilin/dataset-construct/segmentation/DETR/top{TOP_K}/masked_topk/"
OUTPUT_NOMASK_DIR = f"/home/yilin/dataset-construct/segmentation/DETR/top{TOP_K}/no_masked/"



os.makedirs(OUTPUT_MASKED_DIR, exist_ok=True)
os.makedirs(OUTPUT_NOMASK_DIR, exist_ok=True)

# ==========================
# 🧠 初始化 DETR
# ==========================
print("🚀 Loading DETR model...")
processor = DetrImageProcessor.from_pretrained(MODEL_NAME, use_safetensors=True)
model = DetrForSegmentation.from_pretrained(MODEL_NAME, use_safetensors=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print("✅ DETR model loaded on", device)

# ==========================
# 🧰 遮挡 Top-K 函数
# ==========================
def mask_topk_confidence(image_path, topk=TOP_K):
    full_path = os.path.join(IMAGE_DIR, image_path)

    if not os.path.exists(full_path):
        print(f"⚠️ 图像不存在：{full_path}")
        return

    # 读取图像
    img = Image.open(full_path).convert("RGB")
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

    # 没检测到目标
    if len(segments_info) == 0:
        out_path = os.path.join(OUTPUT_NOMASK_DIR, "no_masked_" + image_path)
        cv2.imwrite(out_path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
        print(f"⚠️ 未检测到目标，已保存原图：{out_path}")
        return

    # Top-K segment
    segments_sorted = sorted(segments_info, key=lambda x: x["score"], reverse=True)
    topk_segments = segments_sorted[:topk]

    # 构建mask
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for seg in topk_segments:
        seg_mask = (segmentation == seg["id"]).astype(np.uint8) * 255
        final_mask = cv2.bitwise_or(final_mask, seg_mask)

    # 反转 mask 只保留背景
    final_mask_inv = cv2.bitwise_not(final_mask)
    masked_img = cv2.bitwise_and(np_img, np_img, mask=final_mask_inv)

    # 保存
    out_path = os.path.join(OUTPUT_MASKED_DIR, f"masked_top{topk}_" + image_path)
    cv2.imwrite(out_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

    # Debug 信息
    scores_str = ", ".join([f"{s['score']:.2f}" for s in topk_segments])
    print(f"✅ {image_path}: Top-{topk} 目标已遮挡 | scores={scores_str} | saved={out_path}")


with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📊 共 {len(data)} 张图片待处理")

for item in tqdm(data, desc="🪄 Processing images", ncols=100):
    image_path = item["image"]
    mask_topk_confidence(image_path, topk=TOP_K)


print("🏁 所有图片处理完成！")
