
image_folder="/home/yilin/dataset/train2014/"
data_path="/home/yilin/dataset-construct/preference_data/pref_data.json"
# data_path="/home/yilin/dataset-construct/preference_data/pref_data.json"
pretrained_model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m"
similarity_type="last_hidden_state"


# python -m debugpy --connect 5679 main.py \
python main.py \
    --pretrained_model_name $pretrained_model_name \
    --data_path $data_path \
    --output_dir "./output/$similarity_type/" \
    --image_folder $image_folder \
    --similarity_type $similarity_type \
