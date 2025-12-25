import os
import json
from glob import glob
"""
将 LabelMe JSON 格式转换为 YOLO txt 格式
"""
# ==== 配置区域 ====
json_dir = r"D:\DataBase\coco\train\json_labels"   # LabelMe JSON 文件夹
img_dir = r"D:\DataBase\coco\train\images" # 图片文件夹（可选，用于获取图片大小）
save_dir = r"D:\DataBase\coco\train\labels" # 输出 YOLO txt 文件夹
classes = ['person']  # 按顺序填写类别名称
# =================

os.makedirs(save_dir, exist_ok=True)

json_files = glob(os.path.join(json_dir, '*.json'))

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in classes:
            continue
        class_id = classes.index(label)

        points = shape['points']
        # 如果是矩形，可以使用 min/max 转成 bbox
        if shape['shape_type'] == 'rectangle':
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        # 如果是多边形，用点集方式
        elif shape['shape_type'] == 'polygon':
            # 将坐标归一化
            norm_points = []
            for x, y in points:
                norm_points.append(f"{x/img_width:.6f}")
                norm_points.append(f"{y/img_height:.6f}")
            yolo_lines.append(f"{class_id} " + " ".join(norm_points))

    # 保存 txt
    txt_file = os.path.join(save_dir, os.path.basename(json_file).replace('.json', '.txt'))
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))

print(f"转换完成，共处理 {len(json_files)} 个 JSON 文件")
