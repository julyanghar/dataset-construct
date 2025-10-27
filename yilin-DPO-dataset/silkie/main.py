import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import argparse
from datasets import Dataset
from datasets import load_dataset
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
import json

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all arguments, will ignore cli arguments if specified")
    parser.add_argument("--pretrained_model_name", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument("--output_dir",
        default=None,
        help="output folder"
    )
    parser.add_argument("--image_folder",
        default=None,
        help="image folder"
    )
    parser.add_argument("--data_path",
        default=None,
        help="data folder"
    )
    parser.add_argument("--similarity_type",
        default="pooler_output",
        help="The way calculate cosine similarity"
    )
    args = parser.parse_args()
    return args



def convert_record(record, idx):
    # return {
    #     "idx": idx,
    #     "image": record.get("image"),
    #     "retrieved_image": None,
    #     "masked_response": None,
    #     "chosen": [
    #         {
    #             "content": record.get("prompt", ""),
    #             "role": "user"
    #         },
    #         {
    #             "content": record.get("chosen", ""),
    #             "role": "assistant"
    #         }
    #     ],
    #     "rejected": [
    #         {
    #             "content": record.get("prompt", ""),
    #             "role": "user"
    #         },
    #         {
    #             "content": record.get("rejected", ""),
    #             "role": "assistant"
    #         }
    #     ],
    #     "img_similarity": None,
    #     "text_similarity": None
    # }
    converted = {
        "idx": idx,  # 可以根据需要生成
        "image": record["img_path"],
        "retrieved_image": None,
        "masked_response": None,
        "chosen": [
            {"content": record["prompt"], "role": "user"},
            {"content": record["chosen"], "role": "assistant"}
        ],
        "rejected": [
            {"content": record["prompt"], "role": "user"},
            {"content": record["rejected"], "role": "assistant"}
        ],
        "img_similarity": None,
        "text_similarity": None
    }
    return converted



def convert_records(input_path, output_path):

    # 读取整个 JSON 文件
    with open(input_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)   # 假设 data 是 list

    # valid_prefixes = ('mcq', 'long', 'short')
    # filtered_data = [
    #     item for item in data
    #     if item.get("image") and item["image"].startswith(valid_prefixes)
    # ]
    # 转换每一条样本
    
    converted = []
    for idx, record in tqdm(enumerate(data), total=len(data), desc="Converting", unit="item"):
        converted.append(convert_record(record, idx))

    # 写出到 JSON 文件
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(converted, fout, ensure_ascii=False, indent=4)

    print(f"✅ 已完成转换，输出保存到：{output_path}")





if __name__ == "__main__":
    args = parse_eval_args()
    input_path = "./preference_data/vlfeedback_llava_10k.json"   # 👈 你的原始文件
    output_path = "./preference_data/converted_vlfeedback_llava_10k.json" # 👈 转换后的目标文件
    convert_records(input_path, output_path)
    