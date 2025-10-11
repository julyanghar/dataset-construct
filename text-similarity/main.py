from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
    parser.add_argument("--data_path",
        default=None,
        help="data folder"
    )
    parser.add_argument("--similarity_type",
        default=None,
        help="The way calculate cosine similarity"
    )
    args = parser.parse_args()
    return args

def process_example():
    # 加载模型
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    sentences = [
        "A man is eating food.",
        "Someone is having a meal."
    ]

    embeddings = model.encode(sentences, normalize_embeddings=True)
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"Similarity: {sim:.4f}")

def get_similarity(model, sentences, **args):
    with torch.inference_mode():
        embeddings = model.encode(sentences, normalize_embeddings=True)
        cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # 仅需返回单个元素
    return cos_sim.item()

def process_dataset(model, dataset, **args):
    new_data = []
    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Adding cosine_similarity"):
        data = dict(data)
        chosen_response = data['chosen'][1]['content']
        rejected_response = data['rejected'][1]['content']
        sentences = [chosen_response, rejected_response]
        data["text_similarity"] = get_similarity(model, 
                                        sentences,
                                        **args
                                    )
        new_data.append(data)

    return new_data


if __name__ == "__main__":
    args = parse_eval_args()
    dataset = load_dataset("json", data_files=args.data_path)["train"] # select(range(10)) 
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    new_data = process_dataset(model, dataset, **vars(args))
    filename = os.path.basename(args.data_path)
    output_path = args.output_dir + filename

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"已创建目录: {args.output_dir}")
    else:
        print(f"目录已存在: {args.output_dir}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

