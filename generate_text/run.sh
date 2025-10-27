
image_folder="/home/yilin/dataset/train2014/"
# data_path="/home/yilin/dataset-construct/preference_data/yilin_pref_data_last_hidden_state.json"
data_path="/home/yilin/dataset-construct/preference_data/yilin_pref_data_pooler_output.json"
# data_path="/home/yilin/dataset-construct/preference_data/pref_data.json"

# python -m debugpy --connect 5679 new_unpreferred_response.py \
python new_unpreferred_response.py \
    --data_path $data_path \
    --output_dir "./output/" \
    --image_folder $image_folder \
