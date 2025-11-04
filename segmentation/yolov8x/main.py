import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
import os

model = YOLO("/home/yilin/dataset-construct/segmentation/yolov8x-seg.pt")

input_path = "/home/yilin/dataset-construct/preference_data/converted_vlfeedback_llava_10k.json"
image_dir = "/home/yilin/yilin-DPO/dataset/silkie/merged_images/"

def mask_img(input_path):
    dir_path, filename = os.path.split(input_path)
    output_path = "/home/yilin/dataset-construct/segmentation/" + dir_path + '/' + "masked_" + filename  # 输出文件名
    img = cv2.imread(image_dir + input_path)
    h, w = img.shape[:2]

    results = model(image_dir + input_path)

    res = results[0]

    final_mask = np.zeros((h, w), dtype=np.uint8)
    if res.masks is not None and len(res.masks.xy) > 0:
        points = res.masks.xy[0].astype(np.int32)

        cv2.fillPoly(final_mask, [points], 255)

        final_mask = cv2.bitwise_not(final_mask)

        masked_img = cv2.bitwise_and(img, img, mask=final_mask)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, masked_img)
        print(f"✅ 已保存 masked 图像到: {output_path}")



def mask_img_mdpo(input_path):
    output_path = "/home/yilin/dataset-construct/segmentation/masked/" + "masked_" + input_path  # 输出文件名
    img = cv2.imread(image_dir + input_path)
    h, w = img.shape[:2]

    results = model(image_dir + input_path)

    res = results[0]

    final_mask = np.zeros((h, w), dtype=np.uint8)
    if res.masks is not None and len(res.masks.xy) > 0:
        points = res.masks.xy[0].astype(np.int32)

        cv2.fillPoly(final_mask, [points], 255)

        final_mask = cv2.bitwise_not(final_mask)

        masked_img = cv2.bitwise_and(img, img, mask=final_mask)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, masked_img)
        print(f"✅ 已保存 masked 图像到: {output_path}")
    else:
        output_path = "/home/yilin/dataset-construct/segmentation/no_masked/" + "no_masked_" + input_path  # 输出文件名
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"✅ 已保存 masked 图像到: {output_path}")


with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    mask_img_mdpo(item["image"])
    # 抽取所有 content 文本









