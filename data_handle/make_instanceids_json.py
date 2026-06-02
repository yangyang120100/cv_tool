import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils


# ============================================================
# 路径配置
# ============================================================

IMAGE_DIR = r"D:\DataSets\goose\images\train"
INSTANCE_DIR = r"D:\DataSets\goose\instanceids"
SEMANTIC_DIR = r"D:\DataSets\goose\semantic\train"
OUTPUT_JSON = r"D:\DataSets\goose\annotations\instances_t2.json"


# ============================================================
# Goose 语义类别定义
# 这里保留 thing 类，作为 instance json 的 categories
# category_id 必须和语义 mask 中的 id 一致
# ============================================================

ALL_CATEGORIES = [
    {'id': 0, 'isthing': 0, 'name': 'undefined'},
    {'id': 1, 'isthing': 1, 'name': 'traffic_cone'},
    {'id': 2, 'isthing': 0, 'name': 'snow'},
    {'id': 3, 'isthing': 0, 'name': 'cobble'},
    {'id': 4, 'isthing': 0, 'name': 'obstacle'},
    {'id': 5, 'isthing': 0, 'name': 'leaves'},
    {'id': 6, 'isthing': 1, 'name': 'street_light'},
    {'id': 7, 'isthing': 0, 'name': 'bikeway'},
    {'id': 8, 'isthing': 0, 'name': 'ego_vehicle'},
    {'id': 9, 'isthing': 0, 'name': 'pedestrian_crossing'},
    {'id': 10, 'isthing': 1, 'name': 'road_block'},
    {'id': 11, 'isthing': 0, 'name': 'road_marking'},
    {'id': 12, 'isthing': 1, 'name': 'car'},
    {'id': 13, 'isthing': 1, 'name': 'bicycle'},
    {'id': 14, 'isthing': 1, 'name': 'person'},
    {'id': 15, 'isthing': 1, 'name': 'bus'},
    {'id': 16, 'isthing': 0, 'name': 'forest'},
    {'id': 17, 'isthing': 0, 'name': 'bush'},
    {'id': 18, 'isthing': 0, 'name': 'moss'},
    {'id': 19, 'isthing': 1, 'name': 'traffic_light'},
    {'id': 20, 'isthing': 1, 'name': 'motorcycle'},
    {'id': 21, 'isthing': 0, 'name': 'sidewalk'},
    {'id': 22, 'isthing': 0, 'name': 'curb'},
    {'id': 23, 'isthing': 0, 'name': 'asphalt'},
    {'id': 24, 'isthing': 0, 'name': 'gravel'},
    {'id': 25, 'isthing': 1, 'name': 'boom_barrier'},
    {'id': 26, 'isthing': 0, 'name': 'rail_track'},
    {'id': 27, 'isthing': 0, 'name': 'tree_crown'},
    {'id': 28, 'isthing': 1, 'name': 'tree_trunk'},
    {'id': 29, 'isthing': 0, 'name': 'debris'},
    {'id': 30, 'isthing': 0, 'name': 'crops'},
    {'id': 31, 'isthing': 0, 'name': 'soil'},
    {'id': 32, 'isthing': 1, 'name': 'rider'},
    {'id': 33, 'isthing': 1, 'name': 'animal'},
    {'id': 34, 'isthing': 1, 'name': 'truck'},
    {'id': 35, 'isthing': 1, 'name': 'on_rails'},
    {'id': 36, 'isthing': 1, 'name': 'caravan'},
    {'id': 37, 'isthing': 1, 'name': 'trailer'},
    {'id': 38, 'isthing': 0, 'name': 'building'},
    {'id': 39, 'isthing': 0, 'name': 'wall'},
    {'id': 40, 'isthing': 1, 'name': 'rock'},
    {'id': 41, 'isthing': 0, 'name': 'fence'},
    {'id': 42, 'isthing': 0, 'name': 'guard_rail'},
    {'id': 43, 'isthing': 0, 'name': 'bridge'},
    {'id': 44, 'isthing': 0, 'name': 'tunnel'},
    {'id': 45, 'isthing': 1, 'name': 'pole'},
    {'id': 46, 'isthing': 1, 'name': 'traffic_sign'},
    {'id': 47, 'isthing': 1, 'name': 'misc_sign'},
    {'id': 48, 'isthing': 0, 'name': 'barrier_tape'},
    {'id': 49, 'isthing': 1, 'name': 'kick_scooter'},
    {'id': 50, 'isthing': 0, 'name': 'low_grass'},
    {'id': 51, 'isthing': 0, 'name': 'high_grass'},
    {'id': 52, 'isthing': 0, 'name': 'scenery_vegetation'},
    {'id': 53, 'isthing': 0, 'name': 'sky'},
    {'id': 54, 'isthing': 0, 'name': 'water'},
    {'id': 55, 'isthing': 0, 'name': 'wire'},
    {'id': 56, 'isthing': 0, 'name': 'outlier'},
    {'id': 57, 'isthing': 1, 'name': 'heavy_machinery'},
    {'id': 58, 'isthing': 1, 'name': 'container'},
    {'id': 59, 'isthing': 0, 'name': 'hedge'},
    {'id': 60, 'isthing': 1, 'name': 'barrel'},
    {'id': 61, 'isthing': 0, 'name': 'pipe'},
    {'id': 62, 'isthing': 0, 'name': 'tree_root'},
    {'id': 63, 'isthing': 1, 'name': 'military_vehicle'},
]

THING_CATEGORIES = [c for c in ALL_CATEGORIES if c["isthing"] == 1]
THING_CAT_IDS = {c["id"] for c in THING_CATEGORIES}
CAT_ID_TO_NAME = {c["id"]: c["name"] for c in ALL_CATEGORIES}

# 最小面积过滤
MIN_AREA = 1

# 语义多数投票时，实例内部的主类别占比必须达到阈值才接受
# 太低说明实例和语义严重不对齐，通常是标注有噪声或配错图
MIN_DOMINANCE_RATIO = 0.5

# 如果实例图是 0/1 二值图，是否按连通域拆分
BINARY_MASK_USE_CONNECTED_COMPONENTS = True


# ============================================================
# 工具函数
# ============================================================

def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def encode_rle(mask: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_mask(path: Path):
    return np.array(Image.open(path))


def get_instance_components(instance_arr: np.ndarray):
    """
    支持两种实例图：
    1) 实例 id 图：0,1,2,3...
    2) 二值图：0/1，按连通域拆分
    """
    uniq = np.unique(instance_arr)

    # 二值图
    if len(uniq) <= 2 and set(uniq.tolist()).issubset({0, 1}):
        if not BINARY_MASK_USE_CONNECTED_COMPONENTS:
            return [(instance_arr > 0).astype(bool)]
        binary = (instance_arr > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        comps = []
        for lid in range(1, num_labels):
            comp = labels == lid
            if comp.sum() > 0:
                comps.append(comp)
        return comps

    # 实例 id 图
    comps = []
    for inst_id in uniq:
        if int(inst_id) == 0:
            continue
        comp = instance_arr == inst_id
        if comp.sum() > 0:
            comps.append(comp)
    return comps


def infer_category_from_semantic(semantic: np.ndarray, instance_mask: np.ndarray):
    """
    用实例区域内的语义类别多数投票推断 category_id。
    返回: (category_id, dominance_ratio)
    """
    vals = semantic[instance_mask]
    if vals.size == 0:
        return None, 0.0

    vals = vals.astype(np.int64)
    vals = vals[vals >= 0]
    if vals.size == 0:
        return None, 0.0

    counts = np.bincount(vals)
    category_id = int(np.argmax(counts))
    dominance_ratio = float(counts[category_id]) / float(vals.size)
    return category_id, dominance_ratio


# ============================================================
# 主流程
# ============================================================

def main():
    image_dir = Path(IMAGE_DIR)
    instance_dir = Path(INSTANCE_DIR)
    semantic_dir = Path(SEMANTIC_DIR)
    output_json = Path(OUTPUT_JSON)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # 建立 stem -> mask 路径索引
    instance_map = {p.stem: p for p in instance_dir.glob("*.png")}
    semantic_map = {p.stem: p for p in semantic_dir.glob("*.png")}

    # 原图文件
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(image_dir.glob(ext))
    image_files = sorted(image_files)

    images = []
    annotations = []
    ann_id = 1

    # 只保留 thing 类作为 instance json 的 categories
    categories = [
        {
            "id": int(c["id"]),
            "name": c["name"],
            "supercategory": "thing"
        }
        for c in THING_CATEGORIES
    ]
    category_id_set = {c["id"] for c in categories}

    for image_id, img_path in enumerate(tqdm(image_files, desc="Converting"), start=1):
        stem = img_path.stem

        inst_path = instance_map.get(stem)
        sem_path = semantic_map.get(stem)

        if inst_path is None:
            print(f"[skip] missing instance mask: {img_path.name}")
            continue
        if sem_path is None:
            print(f"[skip] missing semantic mask: {img_path.name}")
            continue

        image = Image.open(img_path)
        width, height = image.size

        instance_arr = load_mask(inst_path)
        semantic_arr = load_mask(sem_path)

        if instance_arr.shape != semantic_arr.shape:
            raise ValueError(
                f"shape mismatch for {img_path.name}: "
                f"instance={instance_arr.shape}, semantic={semantic_arr.shape}"
            )

        images.append({
            "id": int(image_id),
            "width": int(width),
            "height": int(height),
            "file_name": img_path.name
        })

        instance_components = get_instance_components(instance_arr)

        for comp in instance_components:
            area = int(comp.sum())
            if area < MIN_AREA:
                continue

            category_id, dominance_ratio = infer_category_from_semantic(semantic_arr, comp)

            if category_id is None:
                continue

            # 只保留 thing 类
            if category_id not in category_id_set:
                continue

            # 类别投票太弱则跳过，避免语义与实例错配
            if dominance_ratio < MIN_DOMINANCE_RATIO:
                continue

            bbox = mask_to_bbox(comp)
            rle = encode_rle(comp.astype(np.uint8))

            annotations.append({
                "id": int(ann_id),
                "image_id": int(image_id),
                "category_id": int(category_id),
                "segmentation": rle,
                "area": int(area),
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

    out = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print("===================================")
    print(f"saved to: {OUTPUT_JSON}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(annotations)}")
    print(f"categories: {len(categories)}")
    print("===================================")


if __name__ == "__main__":
    main()