import json
from tqdm import tqdm

input_path = "/home/yilin/dataset-construct/yilin-DPO-dataset/silkie/vlfeedback_80k.jsonl"
output_path = "/home/yilin/dataset-construct/yilin-DPO-dataset/silkie/vlfeedback_80k.json"

data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading JSONL"):
        data.append(json.loads(line))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成: {output_path}")
