

import json
import os
import shutil

def copy(input_path, image_path, output_path):
    with open(input_path, 'r') as infile:
        data = json.load(infile)

    for item in data:
        image = item["image"]
        retrieved = item["retrieved_image"]
        masked = item["masked_image"]

        for img in [image, retrieved, masked]:
            source_path = os.path.join(image_path, img)
            target_path = os.path.join(output_path, img)
            if os.path.exists(target_path):
                print("文件已存在:", img)
            else:
                shutil.copy(source_path, target_path)



if __name__ == "__main__":
    input_path = "/home/yilin/dataset-construct/preference_data/masked_pooler.json"
    image_path = "/home/yilin/dataset/train2014"
    output_path = "/home/yilin/dataset/new"
    copy(input_path, image_path, output_path)
       



