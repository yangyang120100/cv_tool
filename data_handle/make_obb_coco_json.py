import json
from pathlib import Path
from tqdm import tqdm

# ========= 1. 核心修复：获取磁盘真实文件名 =========
def get_real_image_name(json_data, json_file, img_dir):
    img_dir = Path(img_dir)
    stem = json_file.stem

    labelme_image_path = json_data.get("imagePath")
    if labelme_image_path:
        raw_name = Path(labelme_image_path).name
        if (img_dir / raw_name).exists():
            return raw_name

    for p in img_dir.iterdir():
        if p.stem == stem and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
            return p.name
    return None

# ========= 2. 辅助函数：计算面积与映射 =========
def load_class_map(txt_path):
    class_map = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name: class_map[name] = i + 1
    return class_map

def polygon_area(points):
    """使用鞋带公式计算多边形面积"""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

# ========= 3. 主转换逻辑 =========
def convert_labelme_obb_to_coco(json_dir, img_dir, classes_txt, output_json):
    json_dir, img_dir = Path(json_dir), Path(img_dir)
    CLASS_MAP = load_class_map(classes_txt)

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    json_files = sorted(list(json_dir.glob("*.json")))

    for json_file in tqdm(json_files, desc="Converting"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_name = get_real_image_name(data, json_file, img_dir)
        if file_name is None:
            continue

        width, height = data.get("imageWidth"), data.get("imageHeight")
        if not width or not height: continue

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        for shape in data.get("shapes", []):
            label = shape["label"]
            if label not in CLASS_MAP: continue

            points = shape["points"] # 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if len(points) < 3: continue

            # --- 核心优化点 ---
            # 1. 处理 segmentation (格式: [[x1, y1, x2, y2, ...]])
            seg_points = [float(p) for pt in points for p in pt]

            # 2. 计算 bbox (格式: [x_min, y_min, width, height])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox = [
                round(float(min_x), 8),
                round(float(min_y), 8),
                round(float(max_x - min_x), 8),
                round(float(max_y - min_y), 8)
            ]

            # 3. 计算面积
            area = polygon_area(points)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_MAP[label],
                "segmentation": [seg_points], # 嵌套列表格式
                "area": round(area, 2),
                "bbox": seg_points,                 # 标准水平框
                "iscrowd": 0
            })
            ann_id += 1
        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in CLASS_MAP.items()]
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)

    print(f"\n成功！共处理 {len(images)} 张图片，标注已存至: {output_json}")

if __name__ == "__main__":
    # 配置路径
    convert_labelme_obb_to_coco(
        json_dir=r"D:\DataBase\Insulator_datas\jsons",
        img_dir=r"D:\DataBase\Insulator_datas\images",
        classes_txt=r"D:\DataBase\Insulator_datas\classes.txt",
        output_json=r"D:\DataBase\Insulator_datas\instances_trainval2017.json"
    )