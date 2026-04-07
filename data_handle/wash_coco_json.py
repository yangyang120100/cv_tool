import os
import json

json_path = r"D:\DataBase\Insulator_datas\insulator_datas_coco\annotations\instances_val2017.json"
img_dir = r"D:\DataBase\Insulator_datas\insulator_datas_coco\val2017"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

valid_images = []
valid_ids = set()

# 过滤不存在图片
for img in data["images"]:
    path = os.path.join(img_dir, img["file_name"])
    if os.path.exists(path):
        valid_images.append(img)
        valid_ids.add(img["id"])
    else:
        print("REMOVE:", img["file_name"])

# 过滤 annotation
valid_annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] in valid_ids
]

data["images"] = valid_images
data["annotations"] = valid_annotations

# 保存
out_path = json_path.replace(".json", "_clean.json")
with open(out_path, "w", encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print(f"Cleaned JSON saved to: {out_path}")