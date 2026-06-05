import json

# ======= 请根据你的实际文件路径修改以下两个变量 =======
input_json  = r"D:\DataBase\road\goose\val\labels\annotations\panoptic_val.json"      # 原始 JSON 文件
output_json = r"D:\DataBase\road\goose\val\labels\annotations\re_panoptic_val.json" # 修改后保存的文件（可覆盖原文件）
# ======================================================

with open(input_json, 'r') as f:
    data = json.load(f)

for img_info in data["images"]:
    if img_info["file_name"].endswith(".png"):
        img_info["file_name"] = img_info["file_name"][:-4] + ".jpg"

with open(output_json, 'w') as f:
    json.dump(data, f, indent=2)  # indent 可以让输出更可读，若不需要可设为 None

print(f"修改完成，已保存至 {output_json}")