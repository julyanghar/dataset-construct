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

def process_example():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)

    pretrained_model_name = args.pretrained_model_name
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
    )

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    print("Pooled output shape:", pooled_output.shape)

def get_similarity(model, image_folder, image_file, retrieved_image_file, **args):
    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    retrieved_image = Image.open(os.path.join(image_folder, retrieved_image_file)).convert('RGB')
    input_image = processor(images=image, return_tensors="pt").to(model.device)
    input_retrieved_image = processor(images=retrieved_image, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        output_image = model(**input_image)
        output_input_retrieved_image = model(**input_retrieved_image)
    
    if args["similarity_type"] == "pooler_output":
        latent_image = output_image.pooler_output
        latent_retrieved_image = output_input_retrieved_image.pooler_output
    # print("Pooled output shape:", latent_image.shape)
    
    # latent_image = output_image.last_hidden_state.mean(dim=1)
    # latent_retrieved_image = output_input_retrieved_image.last_hidden_state.mean(dim=1)

    # latent_image = output_image.last_hidden_state[:,1:,:].mean(dim=1)
    # latent_retrieved_image = output_input_retrieved_image.last_hidden_state[:,1:,:].mean(dim=1)


    cos_sim = F.cosine_similarity(latent_image, latent_retrieved_image, dim=1, eps=1e-8)

    # 仅需返回单个元素
    return cos_sim.item()




def process_dataset(model, dataset, image_folder, **args):
    new_data = []
    for idx, data in tqdm(enumerate(dataset), total=len(dataset), desc="Adding cosine_similarity"):
        data = dict(data)
        data["cosine_similarity"] = get_similarity(model, 
                                        image_folder, 
                                        data["image"], 
                                        data["retrieved_image"],
                                        **args
                                    )
        new_data.append(data)

    new_dataset = Dataset.from_list(new_data)
    return new_dataset

if __name__ == "__main__":
    args = parse_eval_args()
    dataset = load_dataset("json", data_files=args.data_path)["train"] # select(range(10))

    print(dataset[0])

    pretrained_model_name = args.pretrained_model_name
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
    )
    
    
    processed_dataset = process_dataset(model, dataset, **vars(args))

    processed_dataset.to_json(args.output_dir+"yilin_pref_data.jsonl", orient="records", lines=True, force_ascii=False)

    