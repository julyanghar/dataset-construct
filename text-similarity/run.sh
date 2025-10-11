
# data_path="/home/yilin/dataset-construct/preference_data/pref_data.json"
# data_path="/home/yilin/dataset-construct/preference_data/yilin_pref_data_last_hidden_state.json"
data_path="/home/yilin/dataset-construct/preference_data/yilin_pref_data_pooler_output.json"


pretrained_model_name="sentence-transformers/all-mpnet-base-v2"



# python -m debugpy --connect 5679 main.py \
python main.py \
    --pretrained_model_name $pretrained_model_name \
    --data_path $data_path \
    --output_dir "./output/" \


