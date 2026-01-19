import os
import json
import cv2
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
YOLO txt → LabelMe JSON（多线程 & 完全兼容版）

支持：
1) bbox: class xc yc w h
2) polygon: class x1 y1 x2 y2 ...
"""

# ===== 配置区域 =====
txt_dir = r"C:\Users\Kedio\Desktop\add_cable\labels"
img_dir = r"C:\Users\Kedio\Desktop\add_cable\images"
save_dir = r"C:\Users\Kedio\Desktop\add_cable\json_2"
classes = ['cable']
num_workers = 8
# ===================

os.makedirs(save_dir, exist_ok=True)
txt_files = glob(os.path.join(txt_dir, "*.txt"))


def process_one(txt_file):
    name = os.path.splitext(os.path.basename(txt_file))[0]

    # 查找图片
    img_path = None
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        p = os.path.join(img_dir, name + ext)
        if os.path.exists(p):
            img_path = p
            break

    if img_path is None:
        return f"⚠️ 找不到图片: {name}"

    img = cv2.imread(img_path)
    if img is None:
        return f"❌ 图片读取失败: {img_path}"

    h, w = img.shape[:2]
    shapes = []

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        if not line.strip():
            continue

        parts = list(map(float, line.split()))
        class_id = int(parts[0])

        if class_id >= len(classes):
            continue

        label = classes[class_id]

        # ===== bbox =====
        if len(parts) == 5:
            _, xc, yc, bw, bh = parts
            x_min = (xc - bw / 2) * w
            y_min = (yc - bh / 2) * h
            x_max = (xc + bw / 2) * w
            y_max = (yc + bh / 2) * h

            shape = {
                "label": label,
                "points": [
                    [x_min, y_min],
                    [x_max, y_max]
                ],
                "shape_type": "rectangle",
                "flags": {}
            }

        # ===== polygon =====
        else:
            coords = parts[1:]
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            points = []
            for i in range(0, len(coords), 2):
                points.append([coords[i] * w, coords[i + 1] * h])

            shape = {
                "label": label,
                "points": points,
                "shape_type": "polygon",
                "flags": {}
            }

        shapes.append(shape)

    # imageData = None（不是 ""）
    labelme_json = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": None,   # ✅ 关键修复点
        "imageHeight": h,
        "imageWidth": w
    }

    save_path = os.path.join(save_dir, name + ".json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(labelme_json, f, indent=2, ensure_ascii=False)

    return f"{name} 转换完成"


# ===== 多线程执行 =====
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_one, txt) for txt in txt_files]
    for future in as_completed(futures):
        print(future.result())

print(f"\n全部完成，共处理 {len(txt_files)} 个文件")
