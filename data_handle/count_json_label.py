import os
import json
from collections import Counter

#统计路径下的所有json标签文件的类别数

def count_labels_in_json_folder(json_dir):
    """
    统计一个文件夹下所有 LabelMe json 文件中的类别数量

    Args:
        json_dir (str): json 文件夹路径

    Returns:
        Counter: {label: count}
    """
    label_counter = Counter()

    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, file)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取失败: {json_path}, 错误: {e}")
            continue

        shapes = data.get("shapes", [])
        for shape in shapes:
            label = shape.get("label")
            if label is not None:
                label_counter[label] += 1

    return label_counter


json_dir = r"D:\DataBase\Transmission_Tower\train\jsons"

label_counts = count_labels_in_json_folder(json_dir)

print("类别统计结果：")
for label, count in label_counts.items():
    print(f"{label}: {count}")

print(f"\n类别总数: {len(label_counts)}")