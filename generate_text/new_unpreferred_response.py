from openai import OpenAI
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
import base64
import time

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
    args = parser.parse_args()
    return args



def generate_responses(dataset, client, args):
    image_folder = args.image_folder
    new_data = []
    filename = "new_pooler_pref_data.jsonl"
    output_path = args.output_dir + filename

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"已创建目录: {args.output_dir}")
    else:
        print(f"目录已存在: {args.output_dir}")

    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="generate new unpreferred response"):
        data = dict(data)
        image_file = data["image"]
        # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image_path = os.path.join(image_folder, image_file)
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        image_url = f"data:image/jpg;base64,{img_base64}"

        question = data["chosen"][0]["content"]
        preferred_response = data["chosen"][1]["content"]
        existing_unpreferred_response = data["rejected"][1]["content"]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an annotator tasked with generating an unpreferred response "
                    "for contrastive learning in Vision-Language Model training. "
                    "The new unpreferred response should be semantically consistent with "
                    "the existing unpreferred response, preserve key visual elements such "
                    "as people, objects, colors, textures, and positions, and be of lower "
                    "quality than the preferred response (i.e., more vague or slightly inaccurate "
                    "but still relevant to the image)."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f""" Construct a new unpreferred response that is semantically consistent with the existing unpreferred response, preserves the key entities, and remains less precise than the preferred response. Output only the new unpreferred response as plain text.
                        Input image: (provided)
                        Question: {question}
                        Preferred response: {preferred_response}
                        Existing unpreferred response:{existing_unpreferred_response} """},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        data["rejected"][1]["content"] = response.choices[0].message.content
        print(f"""
            preferred_response: {preferred_response}
                ---
            existing_unpreferred_response: {existing_unpreferred_response}
                ---
            new_unpreferred_response: {data["rejected"][1]["content"]}
            --------------------------------------------------
                """)
        new_data.append(data)
        time.sleep(5)
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # new_dataset = Dataset.from_list(new_data)
    return new_data


if __name__ == "__main__":
    args = parse_eval_args()
    dataset = load_dataset("json", data_files=args.data_path)["train"] # select(range(10))
    last_end_index = 151
    dataset = dataset.select(range(last_end_index, len(dataset)))
    # print(dataset[0])

    client = OpenAI()
    new_data = generate_responses(dataset, client, args)
    filename = os.path.basename(args.data_path)
    output_path = args.output_dir + filename

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"已创建目录: {args.output_dir}")
    else:
        print(f"目录已存在: {args.output_dir}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


