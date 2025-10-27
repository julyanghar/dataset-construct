import json
import matplotlib.pyplot as plt


input_file = "/home/yilin/dataset-construct/text-similarity/output/new_yilin_pooler_pref_data.json"
# input_file = "/home/yilin/dataset-construct/preference_data/yilin_pref_data_pooler_output.json"



scores = []
with open(input_file, "r") as f:
    data = json.load(f)
    for item in data:
        # scores.append(item["img_similarity"])
        scores.append(item["text_similarity"])

plt.hist(scores, bins=100, edgecolor='black')
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("./hist/similarity_text.png", dpi=500)   # 或者 "score_histogram.pdf"
# plt.savefig("./output/similarity_last_hidden_state.png", dpi=300)   # 或者 "score_histogram.pdf"


# 可选：不显示图形，适合批处理脚本
# plt.close()


print("✅ 已保存直方图为 similarity_pooler_output.png")
# print("✅ 已保存直方图为 similarity_last_hidden_state.png")