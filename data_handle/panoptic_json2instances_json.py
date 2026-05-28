import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# =========================
# 修改为你的路径
# =========================

PANOPTIC_JSON = r"D:\DataSets\goose\annotations\panoptic_train.json"

PANOPTIC_ROOT = r"D:\DataSets\goose\panoptic\tain"

OUTPUT_JSON = r"D:\DataSets\goose\annotations\instances_train.json"


# =========================
# thing 类原始 category id
# 必须与你 panoptic json 中 categories.id 一致
# =========================

THING_CLASSES_ID = [
    1, 6, 10, 12, 13, 14, 15, 19, 20, 25,
    28, 32, 33, 34, 35, 36, 37, 40, 45, 46,
    47, 49, 57, 58, 60, 63
]


# =========================
# 可选：category name
# 用于生成 categories
# =========================

THING_CLASSES = [
    'traffic_cone',
    'street_light',
    'road_block',
    'car',
    'bicycle',
    'person',
    'bus',
    'traffic_light',
    'motorcycle',
    'boom_barrier',
    'tree_trunk',
    'rider',
    'animal',
    'truck',
    'on_rails',
    'caravan',
    'trailer',
    'rock',
    'pole',
    'traffic_sign',
    'misc_sign',
    'kick_scooter',
    'heavy_machinery',
    'container',
    'barrel',
    'military_vehicle'
]


# =========================
# RGB -> segment id
# COCO panoptic 标准
# =========================

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        return (
            color[:, :, 0].astype(np.int32)
            + color[:, :, 1].astype(np.int32) * 256
            + color[:, :, 2].astype(np.int32) * 256 * 256
        )
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


# =========================
# 读取 panoptic json
# =========================

with open(PANOPTIC_JSON, "r") as f:
    panoptic_data = json.load(f)


# =========================
# categories
# instance json 只保留 thing
# =========================

categories = []

for cid, cname in zip(THING_CLASSES_ID, THING_CLASSES):
    categories.append({
        "id": cid,
        "name": cname,
        "isthing": 1
    })


# =========================
# images
# 直接复用
# =========================

images = panoptic_data["images"]


# =========================
# annotations
# =========================

annotations = []

ann_id = 1


for ann in panoptic_data["annotations"]:

    image_id = ann["image_id"]

    panoptic_file = ann["file_name"]

    panoptic_path = os.path.join(PANOPTIC_ROOT, panoptic_file)

    print(f"processing: {panoptic_path}")

    # -------------------------
    # 读取 panoptic png
    # -------------------------

    panoptic_png = np.array(Image.open(panoptic_path), dtype=np.uint8)

    panoptic_id_map = rgb2id(panoptic_png)

    # -------------------------
    # 遍历 segments
    # -------------------------

    for segment in ann["segments_info"]:

        category_id = segment["category_id"]

        # 只保留 thing
        if category_id not in THING_CLASSES_ID:
            continue

        segment_id = segment["id"]

        # -------------------------
        # 提取 mask
        # -------------------------

        mask = panoptic_id_map == segment_id

        if mask.sum() == 0:
            continue

        mask = mask.astype(np.uint8)

        # -------------------------
        # RLE 编码
        # pycocotools 要求 Fortran order
        # -------------------------

        rle = mask_utils.encode(
            np.asfortranarray(mask)
        )

        rle["counts"] = rle["counts"].decode("utf-8")

        # -------------------------
        # bbox
        # COCO 格式：
        # [x, y, w, h]
        # -------------------------

        ys, xs = np.where(mask)

        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        bbox = [
            x_min,
            y_min,
            x_max - x_min + 1,
            y_max - y_min + 1
        ]

        # -------------------------
        # annotation
        # -------------------------

        coco_ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "area": int(mask.sum()),
            "bbox": bbox,
            "iscrowd": segment.get("iscrowd", 0)
        }

        annotations.append(coco_ann)

        ann_id += 1


# =========================
# 最终 instance json
# =========================

instance_json = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}


# =========================
# 保存
# =========================

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(instance_json, f)

print()
print("===================================")
print(f"saved to: {OUTPUT_JSON}")
print(f"images: {len(images)}")
print(f"annotations: {len(annotations)}")
print("===================================")