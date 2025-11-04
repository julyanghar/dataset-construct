import json
import re

# 中文字符的正则
pattern = re.compile(r'[\u4e00-\u9fa5]')

input_path = "/home/yilin/dataset-construct/yilin-DPO-dataset/add_masked_img/preference_data/yilin_pref_data_pooler_output.json"
output_path = "/home/yilin/dataset-construct/yilin-DPO-dataset/add_masked_img/preference_data/masked_pooler.json"

# 读取数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = []
for item in data:
    item["masked_image"] = "masked_top2_" + item["image"]
    filtered_data.append(item)

for new_idx, item in enumerate(filtered_data):
    item["idx"] = new_idx

print(f"原始数据量: {len(data)}")
print(f"过滤后数据量: {len(filtered_data)}")

# 写回新文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"✅ 过滤完成，已保存到 {output_path}")
