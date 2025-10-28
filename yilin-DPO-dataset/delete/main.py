import os

# 📁 目标文件夹路径
folder_path = "/home/yilin/dataset/train2014"

# 🧹 想要删除的文件名前缀
prefix = "masked_top2_"

delete_count = 0
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.startswith(prefix):   # ✅ 判断前缀
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)      # 🧨 删除文件
            delete_count += 1
            print(f"✅ 已删除: {file_path}")
        except Exception as e:
            print(f"❌ 删除失败 {file_path}: {e}")

print(f"🏁 删除完成，共删除 {delete_count} 个文件。")
