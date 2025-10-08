
image_folder="/home/yilin/dataset/train2014/"
data_path="./preference_data/pref_data.json"
# data_path="/home/yilin/dataset-construct/output/yilin_pref_data.json"
pretrained_model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m"

# python -m debugpy --connect 5679 dinov3.py \
python dinov3.py \
    --pretrained_model_name $pretrained_model_name \
    --data_path $data_path \
    --output_dir "./output/" \
    --image_folder $image_folder \
