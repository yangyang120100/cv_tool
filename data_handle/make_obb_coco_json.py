import os
import json
from pathlib import Path
from tqdm import tqdm

def convert_labelme_obb_to_coco(json_dir, output_json):
    json_dir = Path(json_dir)

    images = []
    annotations = []
    categories = {}
    category_id = 1

    ann_id = 1
    img_id = 1

    for json_file in tqdm(list(json_dir.glob("*.json"))):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # file_name = data["imagePath"]
        file_name = json_file.with_suffix(".jpg").name
        width = data["imageWidth"]
        height = data["imageHeight"]

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            # 类别映射
            if label not in categories:
                categories[label] = category_id
                category_id += 1

            cat_id = categories[label]

            # flatten 8点
            bbox = [p for point in points for p in point]

            # 计算 area（简单多边形面积）
            area = polygon_area(points)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

            ann_id += 1

        img_id += 1

    # categories 转换
    categories_list = [
        {"id": v, "name": k} for k, v in categories.items()
    ]

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list
    }

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Saved to {output_json}")


def polygon_area(points):
    """Shoelace formula"""
    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


if __name__ == "__main__":
    convert_labelme_obb_to_coco(
        json_dir=r"D:\DataBase\Insulator_datas\yolo\jsons",   # 你的json目录
        output_json=r"D:\DataBase\Insulator_datas\yolo\instances_val2017.json"
    )