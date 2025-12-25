import json
import os

"""
将 COCO 格式 annotations 拆分为每张图像一个 JSON，
并转换为 Labelme 格式：
- shapes -> bounding boxes
- imagePath, imageHeight, imageWidth, imageData
"""



def coco_to_labelme(coco_json_path, output_dir, label_name="person"):
    """
    将 COCO 格式 annotations 拆分为每张图像一个 JSON，
    并转换为 Labelme 格式：
    - shapes -> bounding boxes
    - imagePath, imageHeight, imageWidth, imageData
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco.get('images', [])}
    annotations = coco.get('annotations', [])

    # 构建 image_id -> annotations 列表
    img_annos = {}
    for anno in annotations:
        img_id = anno['image_id']
        img_annos.setdefault(img_id, []).append(anno)

    for img_id, img_info in images.items():
        shapes = []
        for anno in img_annos.get(img_id, []):
            x, y, w, h = anno['bbox']
            # 将 COCO bbox 转换为 Labelme rectangle points
            points = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]
            shape = {
                "label": label_name,
                "score": None,
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": []
            }
            shapes.append(shape)

        labelme_json = {
            "version": "2.5.4",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_info["file_name"],
            "imageData": None,
            "imageHeight": img_info["height"],
            "imageWidth": img_info["width"]
        }

        # 输出 JSON 文件，以图像文件名命名
        base_name = os.path.splitext(img_info["file_name"])[0]
        out_path = os.path.join(output_dir, f"{base_name}.json")
        with open(out_path, 'w', encoding='utf-8') as fw:
            json.dump(labelme_json, fw, ensure_ascii=False, indent=2)

    print(f"Done. Generated {len(images)} Labelme JSON files in '{output_dir}'.")


if __name__ == "__main__":
    input_coco = r"D:\DataBase\coco\annotations\train.json"        # 原始 COCO annotations
    output_folder = r"D:\DataBase\coco\train\json_labels"   # 输出目录
    coco_to_labelme(input_coco, output_folder)
