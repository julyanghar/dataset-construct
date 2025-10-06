
image_folder="/data/yilin/train2014/"
data_path="./preference_data/pref_data.json"


deepspeed --include=localhost:2 --master_port 50000 dinov3.py \