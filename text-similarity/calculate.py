import json
import os




if __name__ == "__main__":
    input_file = "/home/yilin/dataset-construct/text-similarity/output/yilin_pref_data_pooler_output.json"
    
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    total = len(data)
    threshold = 0.9
    count = 0
    for item in data:
        if item["text_similarity"] > threshold:
            count += 1

    print(f"总数：{total}, 超过阈值数：{count}")
            
