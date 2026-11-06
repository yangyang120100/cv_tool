"""
过滤 JSON 标签文件中的指定类别（LabelMe 格式）
适用场景：每个图片有一个 JSON 标注文件，内含 shapes 列表，每个 shape 有 label 字段
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor

labels_dir = r"E:\test_images\jsons"      # 原始 JSON 标签文件夹
out_dir = r"E:\test_images\jsons" # 过滤后的输出文件夹
os.makedirs(out_dir, exist_ok=True)

# 需要删除的类别名称（填写 label 字符串）
remove_labels = {'bicycle','person','conductor','car','work-vehicle','large-vehicle'}# 替换为实际要删除的类别

def process_file(fname):
    if not fname.lower().endswith('.json'):
        return

    in_path = os.path.join(labels_dir, fname)
    out_path = os.path.join(out_dir, fname)

    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败 {fname}: {e}")
        return

    # 校验shapes 字段（LabelMe 格式）
    shapes = data.get('shapes', [])
    if not shapes:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    # 过滤 shapes：保留 label 不在删除列表中的
    new_shapes = [shape for shape in shapes if shape.get('label') not in remove_labels]
    data['shapes'] = new_shapes

    # 写入过滤后的 JSON
    with open(out_path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)

    # 可选：如果过滤后没有任何 shape，可以选择删除输出文件（这里保留空 shapes 文件）

    print(f"处理完成: {fname}, 保留标注数: {len(new_shapes)}")

max_workers = 8
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_file, os.listdir(labels_dir))

print("所有文件处理完毕！")