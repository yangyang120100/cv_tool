import json
from pathlib import Path
from tqdm import tqdm


# ========= 1. 核心修复：获取磁盘真实文件名 =========
def get_real_image_name(json_data, json_file, img_dir):
    """
    1. 优先从 JSON 的 imagePath 获取原始文件名
    2. 如果找不到，则在磁盘上搜索同名文件（忽略大小写进行匹配，但返回真实文件名）
    """
    img_dir = Path(img_dir)
    stem = json_file.stem

    # 方案 A: 从 Labelme 数据中直接提取原始路径名
    labelme_image_path = json_data.get("imagePath")
    if labelme_image_path:
        raw_name = Path(labelme_image_path).name
        if (img_dir / raw_name).exists():
            return raw_name

    # 方案 B: 磁盘扫描（解决 .jpg vs .JPG 问题的终极手段）
    # 查找所有与 stem 相同的文件，不区分大小写地检查后缀
    for p in img_dir.iterdir():
        if p.stem == stem and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
            return p.name  # 返回磁盘上实际存储的名字，例如 "ABC.JPG"

    return None


# ========= 2. 其他辅助函数 =========
def load_class_map(txt_path):
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {txt_path}")
    class_map = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name: class_map[name] = i + 1
    return class_map


def polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i];
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

        # 调用修复后的函数
        file_name = get_real_image_name(data, json_file, img_dir)

        if file_name is None:
            print(f"\n[WARN] 找不到图片文件: {json_file.stem}.* 在目录 {img_dir}")
            continue

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height: continue

        images.append({
            "id": img_id,
            "file_name": file_name,  # 这里会严格遵循磁盘上的 DJI_0537_68_crop_1.JPG
            "width": width,
            "height": height
        })

        for shape in data.get("shapes", []):
            label = shape["label"]
            if label not in CLASS_MAP: continue

            points = shape["points"]
            if len(points) != 4: continue

            # === 关键修改：Oriented-DETR 期望 bbox 字段存储的是 8 个坐标值 ===
            # 将 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 展平为 [x1, y1, x2, y2, x3, y3, x4, y4]
            obb_bbox = [float(p) for point in points for p in point]

            # 计算面积和外接矩形（用于兼容性，但主要靠上面的 obb_bbox）
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            area = polygon_area(points)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_MAP[label],
                "bbox": obb_bbox,  # 存入 8 个值，解决 reshape 报错
                "segmentation": [obb_bbox],  # 保持一致
                "area": area,
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

    print(f"\n转换完成！保存至: {output_json}")
    print(f"共计图片: {len(images)} 张")


if __name__ == "__main__":
    convert_labelme_obb_to_coco(
        json_dir=r"D:\DataBase\Fuse_isolating-switch_drop-out-lightning_arrester_lightning-arrester\jsons\val",
        img_dir=r"D:\DataBase\Fuse_isolating-switch_drop-out-lightning_arrester_lightning-arrester\images\val",
        classes_txt=r"D:\DataBase\Fuse_isolating-switch_drop-out-lightning_arrester_lightning-arrester\classes.txt",
        output_json=r"D:\DataBase\Fuse_isolating-switch_drop-out-lightning_arrester_lightning-arrester\insulator_datas_coco\instances_val2017.json"
    )