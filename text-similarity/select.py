import json
import os



def select(input_path, output_path):
    with open(input_path, 'r') as infile:
        data = json.load(infile)
    
    output_dir = os.path.dirname(output_path)

    total = len(data)
    threshold = 0.9
    count = 0
    new_data = []
    for item in data:
        new_item = {}
        if item["text_similarity"] > threshold:
            new_item["chosen"] = item["chosen"][1]["content"]
            new_item["rejected"] = item["rejected"][1]["content"]
            new_item["image"] = item["image"]
            new_item["text_similarity"] = item["text_similarity"]
            new_data.append(new_item)
            count += 1

    print(f"总数：{total}, 超过阈值数：{count}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")
    else:
        print(f"目录已存在: {output_dir}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4) 



def filter(input_path, output_path, threshold=0.9):
    with open(input_path, 'r') as infile:
        data = json.load(infile)
    
    output_dir = os.path.dirname(output_path)

    total = len(data)
    count = 0
    new_data = []
    for item in data:
        if item["text_similarity"] < threshold:
            new_data.append(item)
            count += 1

    print(f"总数：{total}, 低于阈值数：{count}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")
    else:
        print(f"目录已存在: {output_dir}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4) 


if __name__ == "__main__":
    input_path = "/home/yilin/dataset-construct/preference_data/masked_pooler.json"
    output_path = "/home/yilin/dataset-construct/text-similarity/output/filterred.json"
    filter(input_path, output_path, 0.9)

       



