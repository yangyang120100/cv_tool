import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

#生成全景分割标注数据
# ============================================================
# 用户配置
# ============================================================

# labelids 根目录
LABEL_DIR = r"D:\DataBase\road\goose\val\labels\labelids"

# instanceids 根目录
INSTANCE_DIR = r"D:\DataBase\road\goose\val\labels\instanceids"

# 输出 panoptic png 目录
OUTPUT_PANOPTIC_DIR = r"D:\DataBase\road\goose\val\labels\panoptic"

# 输出 json
OUTPUT_JSON = r"D:\DataBase\road\goose\val\labels\panoptic.json"


# ------------------------------------------------------------
# 类别定义
# id 必须与 labelids 中语义id一致
# isthing:
#   1 -> 可实例化目标
#   0 -> stuff
# ------------------------------------------------------------

CATEGORIES = [{'color': (0, 0, 0), 'id': 0, 'isthing': 0, 'name': 'undefined'}, {'color': (0, 255, 255), 'id': 1, 'isthing': 1, 'name': 'traffic_cone'}, {'color': (160, 87, 209), 'id': 2, 'isthing': 0, 'name': 'snow'}, {'color': (255, 52, 255), 'id': 3, 'isthing': 0, 'name': 'cobble'}, {'color': (70, 74, 255), 'id': 4, 'isthing': 0, 'name': 'obstacle'}, {'color': (65, 137, 0), 'id': 5, 'isthing': 0, 'name': 'leaves'}, {'color': (166, 111, 0), 'id': 6, 'isthing': 1, 'name': 'street_light'}, {'color': (89, 0, 163), 'id': 7, 'isthing': 0, 'name': 'bikeway'}, {'color': (229, 219, 255), 'id': 8, 'isthing': 0, 'name': 'ego_vehicle'}, {'color': (0, 73, 122), 'id': 9, 'isthing': 0, 'name': 'pedestrian_crossing'}, {'color': (166, 0, 0), 'id': 10, 'isthing': 1, 'name': 'road_block'}, {'color': (172, 255, 99), 'id': 11, 'isthing': 0, 'name': 'road_marking'}, {'color': (98, 118, 183), 'id': 12, 'isthing': 1, 'name': 'car'}, {'color': (67, 77, 0), 'id': 13, 'isthing': 1, 'name': 'bicycle'}, {'color': (255, 176, 143), 'id': 14, 'isthing': 1, 'name': 'person'}, {'color': (135, 125, 153), 'id': 15, 'isthing': 1, 'name': 'bus'}, {'color': (7, 0, 90), 'id': 16, 'isthing': 0, 'name': 'forest'}, {'color': (147, 150, 128), 'id': 17, 'isthing': 0, 'name': 'bush'}, {'color': (189, 168, 180), 'id': 18, 'isthing': 0, 'name': 'moss'}, {'color': (0, 68, 27), 'id': 19, 'isthing': 1, 'name': 'traffic_light'}, {'color': (1, 198, 79), 'id': 20, 'isthing': 1, 'name': 'motorcycle'}, {'color': (255, 93, 59), 'id': 21, 'isthing': 0, 'name': 'sidewalk'}, {'color': (83, 59, 74), 'id': 22, 'isthing': 0, 'name': 'curb'}, {'color': (128, 47, 255), 'id': 23, 'isthing': 0, 'name': 'asphalt'}, {'color': (90, 97, 97), 'id': 24, 'isthing': 0, 'name': 'gravel'}, {'color': (45, 54, 52), 'id': 25, 'isthing': 1, 'name': 'boom_barrier'}, {'color': (0, 121, 107), 'id': 26, 'isthing': 0, 'name': 'rail_track'}, {'color': (160, 194, 0), 'id': 27, 'isthing': 0, 'name': 'tree_crown'}, {'color': (146, 170, 255), 'id': 28, 'isthing': 1, 'name': 'tree_trunk'}, {'color': (76, 111, 136), 'id': 29, 'isthing': 0, 'name': 'debris'}, {'color': (237, 134, 0), 'id': 30, 'isthing': 0, 'name': 'crops'}, {'color': (0, 97, 209), 'id': 31, 'isthing': 0, 'name': 'soil'}, {'color': (255, 239, 221), 'id': 32, 'isthing': 1, 'name': 'rider'}, {'color': (53, 0, 0), 'id': 33, 'isthing': 1, 'name': 'animal'}, {'color': (75, 79, 123), 'id': 34, 'isthing': 1, 'name': 'truck'}, {'color': (153, 194, 161), 'id': 35, 'isthing': 1, 'name': 'on_rails'}, {'color': (24, 0, 48), 'id': 36, 'isthing': 1, 'name': 'caravan'}, {'color': (216, 166, 10), 'id': 37, 'isthing': 1, 'name': 'trailer'}, {'color': (73, 51, 1), 'id': 38, 'isthing': 0, 'name': 'building'}, {'color': (111, 132, 0), 'id': 39, 'isthing': 0, 'name': 'wall'}, {'color': (1, 33, 55), 'id': 40, 'isthing': 1, 'name': 'rock'}, {'color': (0, 181, 255), 'id': 41, 'isthing': 0, 'name': 'fence'}, {'color': (237, 255, 194), 'id': 42, 'isthing': 0, 'name': 'guard_rail'}, {'color': (191, 121, 160), 'id': 43, 'isthing': 0, 'name': 'bridge'}, {'color': (68, 7, 204), 'id': 44, 'isthing': 0, 'name': 'tunnel'}, {'color': (178, 185, 192), 'id': 45, 'isthing': 1, 'name': 'pole'}, {'color': (153, 255, 194), 'id': 46, 'isthing': 1, 'name': 'traffic_sign'}, {'color': (9, 30, 0), 'id': 47, 'isthing': 1, 'name': 'misc_sign'}, {'color': (89, 196, 190), 'id': 48, 'isthing': 0, 'name': 'barrier_tape'}, {'color': (98, 0, 111), 'id': 49, 'isthing': 1, 'name': 'kick_scooter'}, {'color': (102, 189, 12), 'id': 50, 'isthing': 0, 'name': 'low_grass'}, {'color': (255, 195, 238), 'id': 51, 'isthing': 0, 'name': 'high_grass'}, {'color': (117, 109, 69), 'id': 52, 'isthing': 0, 'name': 'scenery_vegetation'}, {'color': (104, 123, 183), 'id': 53, 'isthing': 0, 'name': 'sky'}, {'color': (161, 135, 122), 'id': 54, 'isthing': 0, 'name': 'water'}, {'color': (0, 140, 255), 'id': 55, 'isthing': 0, 'name': 'wire'}, {'color': (102, 141, 120), 'id': 56, 'isthing': 0, 'name': 'outlier'}, {'color': (159, 208, 250), 'id': 57, 'isthing': 1, 'name': 'heavy_machinery'}, {'color': (154, 138, 255), 'id': 58, 'isthing': 1, 'name': 'container'}, {'color': (23, 211, 232), 'id': 59, 'isthing': 0, 'name': 'hedge'}, {'color': (0, 208, 208), 'id': 60, 'isthing': 1, 'name': 'barrel'}, {'color': (0, 0, 221), 'id': 61, 'isthing': 0, 'name': 'pipe'}, {'color': (132, 164, 196), 'id': 62, 'isthing': 0, 'name': 'tree_root'}, {'color': (64, 64, 64), 'id': 63, 'isthing': 1, 'name': 'military_vehicle'}]

# ============================================================
# 工具函数
# ============================================================


def id2rgb(segment_id):
    """
    COCO panoptic:
    segment_id -> RGB
    """
    r = segment_id % 256
    g = (segment_id // 256) % 256
    b = (segment_id // 65536) % 256
    return [r, g, b]


def rgb2id(color):
    """
    RGB -> segment_id
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        return (
            color[:, :, 0].astype(np.int32)
            + 256 * color[:, :, 1].astype(np.int32)
            + 256 * 256 * color[:, :, 2].astype(np.int32)
        )
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


def compute_bbox(mask):
    """
    mask -> [x, y, w, h]
    """
    ys, xs = np.where(mask)

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    return [
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1)
    ]


# ============================================================
# 主逻辑
# ============================================================

os.makedirs(OUTPUT_PANOPTIC_DIR, exist_ok=True)

category_dict = {
    c["id"]: c
    for c in CATEGORIES
}

images = []
annotations = []

segment_global_id = 1

label_files = sorted(list(Path(LABEL_DIR).glob("*.png")))

for image_id, label_path in enumerate(tqdm(label_files), start=1):

    file_name = label_path.name

    instance_path = Path(INSTANCE_DIR) / file_name

    if not instance_path.exists():
        print(f"missing instance map: {instance_path}")
        continue

    semantic = np.array(Image.open(label_path))
    instance = np.array(Image.open(instance_path))

    h, w = semantic.shape

    # panoptic png
    panoptic_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    segments_info = []

    unique_classes = np.unique(semantic)

    for class_id in unique_classes:

        if class_id not in category_dict:
            continue

        category = category_dict[class_id]

        # ====================================================
        # thing 类
        # ====================================================
        if category["isthing"] == 1:

            class_mask = semantic == class_id

            instance_ids = np.unique(instance[class_mask])

            for inst_id in instance_ids:

                if inst_id == 0:
                    continue

                mask = (
                    (semantic == class_id)
                    & (instance == inst_id)
                )

                if mask.sum() == 0:
                    continue

                seg_id = segment_global_id
                segment_global_id += 1

                color = id2rgb(seg_id)

                panoptic_rgb[mask] = color

                area = int(mask.sum())

                bbox = compute_bbox(mask)

                segments_info.append({
                    "id": int(seg_id),
                    "category_id": int(class_id),
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })

        # ====================================================
        # stuff 类
        # ====================================================
        else:

            mask = semantic == class_id

            if mask.sum() == 0:
                continue

            seg_id = segment_global_id
            segment_global_id += 1

            color = id2rgb(seg_id)

            panoptic_rgb[mask] = color

            area = int(mask.sum())

            bbox = compute_bbox(mask)

            segments_info.append({
                "id": int(seg_id),
                "category_id": int(class_id),
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })

    # 保存 panoptic png
    panoptic_file_name = file_name

    Image.fromarray(panoptic_rgb).save(
        os.path.join(
            OUTPUT_PANOPTIC_DIR,
            panoptic_file_name
        )
    )

    images.append({
        "id": image_id,
        "width": w,
        "height": h,
        "file_name": file_name
    })

    annotations.append({
        "image_id": image_id,
        "file_name": panoptic_file_name,
        "segments_info": segments_info
    })


# ============================================================
# 保存 json
# ============================================================

panoptic_json = {
    "images": images,
    "annotations": annotations,
    "categories": CATEGORIES
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(panoptic_json, f)

print("done")