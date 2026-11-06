"""
保存指定类别的标签文件 根据 annotation 中的 `label` 字段进行过滤"""

import os
import json
from tqdm import tqdm


def save_specify_label_file(json_dir, save_dir, specify_labels):
    """保存只包含指定 label 的标签文件"""

    os.makedirs(save_dir, exist_ok=True)

    for json_name in tqdm(os.listdir(json_dir), desc="Processing json files"):
        if not json_name.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, json_name)
        save_path = os.path.join(save_dir, json_name)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "shapes" in data:
            annotations_key = "shapes"
        elif "annotations" in data:
            annotations_key = "annotations"
        else:
            continue  # 未知结构直接跳过

        annotations = data[annotations_key]

        #过滤 label
        keep_annotations = [
            ann for ann in annotations
            if ann.get("label") in specify_labels
        ]

        if not keep_annotations:
            continue

        data[annotations_key] = keep_annotations

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    json_dir = r'D:\DataBase\Transmission_Tower\train\jsons'
    save_dir = r'D:\DataBase\Transmission_Tower\delet_datas\jsons'
    specify_labels = ['pole_steelpipe', 'pole_jiaogangta']

    save_specify_label_file(json_dir, save_dir, specify_labels)
