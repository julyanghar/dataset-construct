
image_folder="/home/yilin/dataset/train2014/"
# data_path="/home/yilin/dataset-construct/preference_data/yilin_pref_data_last_hidden_state.json"
data_path="/home/yilin/dataset-construct/preference_data/yilin_pref_data_pooler_output.json"
# data_path="/home/yilin/dataset-construct/preference_data/pref_data.json"


export OPENAI_API_KEY="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"

# python -m debugpy --connect 5679 new_unpreferred_response.py \
python new_unpreferred_response.py \
    --data_path $data_path \
    --output_dir "./output/" \
    --image_folder $image_folder \
