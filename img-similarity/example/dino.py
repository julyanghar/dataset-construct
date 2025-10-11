from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import os

# 模型名称
model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# model_name = "facebook/dinov2-small"

# 加载模型与预处理器
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval().cuda()  # 如果有GPU

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

# 测试图片
# image = Image.open("example.jpg").convert("RGB")
image = get_img()

inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    feats = outputs.last_hidden_state[:, 0]  # CLS token，(batch, hidden_dim)
    feats = torch.nn.functional.normalize(feats, dim=-1)
print("特征维度：", feats.shape)
