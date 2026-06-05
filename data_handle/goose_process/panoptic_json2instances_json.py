import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# 路径配置
# ============================================================

# 原图目录（jpg/png 都可以，按你的实际情况改）
IMAGE_DIR = r"D:\DataBase\road\goose\val\images\vis"

# 语义 mask 目录（单通道 png）
SEMANTIC_DIR = r"D:\DataBase\road\goose\val\labels\semantic"

# 实例 mask 目录（单通道 png）
INSTANCE_DIR = r"D:\DataBase\road\goose\val\labels\instanceids"

# 输出 panoptic png 目录
OUTPUT_PANOPTIC_DIR = r"D:\DataBase\road\goose\val\labels\panoptic"

# 输出 panoptic json
OUTPUT_JSON = r"D:\DataBase\road\goose\val\labels\annotations\panoptic_train.json"


# ============================================================
# 类别定义
# id / isthing / name / color
# 建议保持和你现有 Goose 标注一致
# ============================================================

CATEGORIES = [
    {'color': (0, 0, 0), 'id': 0, 'isthing': 0, 'name': 'undefined'},
    {'color': (0, 255, 255), 'id': 1, 'isthing': 1, 'name': 'traffic_cone'},
    {'color': (160, 87, 209), 'id': 2, 'isthing': 0, 'name': 'snow'},
    {'color': (255, 52, 255), 'id': 3, 'isthing': 0, 'name': 'cobble'},
    {'color': (70, 74, 255), 'id': 4, 'isthing': 0, 'name': 'obstacle'},
    {'color': (65, 137, 0), 'id': 5, 'isthing': 0, 'name': 'leaves'},
    {'color': (166, 111, 0), 'id': 6, 'isthing': 1, 'name': 'street_light'},
    {'color': (89, 0, 163), 'id': 7, 'isthing': 0, 'name': 'bikeway'},
    {'color': (229, 219, 255), 'id': 8, 'isthing': 0, 'name': 'ego_vehicle'},
    {'color': (0, 73, 122), 'id': 9, 'isthing': 0, 'name': 'pedestrian_crossing'},
    {'color': (166, 0, 0), 'id': 10, 'isthing': 1, 'name': 'road_block'},
    {'color': (172, 255, 99), 'id': 11, 'isthing': 0, 'name': 'road_marking'},
    {'color': (98, 118, 183), 'id': 12, 'isthing': 1, 'name': 'car'},
    {'color': (67, 77, 0), 'id': 13, 'isthing': 1, 'name': 'bicycle'},
    {'color': (255, 176, 143), 'id': 14, 'isthing': 1, 'name': 'person'},
    {'color': (135, 125, 153), 'id': 15, 'isthing': 1, 'name': 'bus'},
    {'color': (7, 0, 90), 'id': 16, 'isthing': 0, 'name': 'forest'},
    {'color': (147, 150, 128), 'id': 17, 'isthing': 0, 'name': 'bush'},
    {'color': (189, 168, 180), 'id': 18, 'isthing': 0, 'name': 'moss'},
    {'color': (0, 68, 27), 'id': 19, 'isthing': 1, 'name': 'traffic_light'},
    {'color': (1, 198, 79), 'id': 20, 'isthing': 1, 'name': 'motorcycle'},
    {'color': (255, 93, 59), 'id': 21, 'isthing': 0, 'name': 'sidewalk'},
    {'color': (83, 59, 74), 'id': 22, 'isthing': 0, 'name': 'curb'},
    {'color': (128, 47, 255), 'id': 23, 'isthing': 0, 'name': 'asphalt'},
    {'color': (90, 97, 97), 'id': 24, 'isthing': 0, 'name': 'gravel'},
    {'color': (45, 54, 52), 'id': 25, 'isthing': 1, 'name': 'boom_barrier'},
    {'color': (0, 121, 107), 'id': 26, 'isthing': 0, 'name': 'rail_track'},
    {'color': (160, 194, 0), 'id': 27, 'isthing': 0, 'name': 'tree_crown'},
    {'color': (146, 170, 255), 'id': 28, 'isthing': 1, 'name': 'tree_trunk'},
    {'color': (76, 111, 136), 'id': 29, 'isthing': 0, 'name': 'debris'},
    {'color': (237, 134, 0), 'id': 30, 'isthing': 0, 'name': 'crops'},
    {'color': (0, 97, 209), 'id': 31, 'isthing': 0, 'name': 'soil'},
    {'color': (255, 239, 221), 'id': 32, 'isthing': 1, 'name': 'rider'},
    {'color': (53, 0, 0), 'id': 33, 'isthing': 1, 'name': 'animal'},
    {'color': (75, 79, 123), 'id': 34, 'isthing': 1, 'name': 'truck'},
    {'color': (153, 194, 161), 'id': 35, 'isthing': 1, 'name': 'on_rails'},
    {'color': (24, 0, 48), 'id': 36, 'isthing': 1, 'name': 'caravan'},
    {'color': (216, 166, 10), 'id': 37, 'isthing': 1, 'name': 'trailer'},
    {'color': (73, 51, 1), 'id': 38, 'isthing': 0, 'name': 'building'},
    {'color': (111, 132, 0), 'id': 39, 'isthing': 0, 'name': 'wall'},
    {'color': (1, 33, 55), 'id': 40, 'isthing': 1, 'name': 'rock'},
    {'color': (0, 181, 255), 'id': 41, 'isthing': 0, 'name': 'fence'},
    {'color': (237, 255, 194), 'id': 42, 'isthing': 0, 'name': 'guard_rail'},
    {'color': (191, 121, 160), 'id': 43, 'isthing': 0, 'name': 'bridge'},
    {'color': (68, 7, 204), 'id': 44, 'isthing': 0, 'name': 'tunnel'},
    {'color': (178, 185, 192), 'id': 45, 'isthing': 1, 'name': 'pole'},
    {'color': (153, 255, 194), 'id': 46, 'isthing': 1, 'name': 'traffic_sign'},
    {'color': (9, 30, 0), 'id': 47, 'isthing': 1, 'name': 'misc_sign'},
    {'color': (89, 196, 190), 'id': 48, 'isthing': 0, 'name': 'barrier_tape'},
    {'color': (98, 0, 111), 'id': 49, 'isthing': 1, 'name': 'kick_scooter'},
    {'color': (102, 189, 12), 'id': 50, 'isthing': 0, 'name': 'low_grass'},
    {'color': (255, 195, 238), 'id': 51, 'isthing': 0, 'name': 'high_grass'},
    {'color': (117, 109, 69), 'id': 52, 'isthing': 0, 'name': 'scenery_vegetation'},
    {'color': (104, 123, 183), 'id': 53, 'isthing': 0, 'name': 'sky'},
    {'color': (161, 135, 122), 'id': 54, 'isthing': 0, 'name': 'water'},
    {'color': (0, 140, 255), 'id': 55, 'isthing': 0, 'name': 'wire'},
    {'color': (102, 141, 120), 'id': 56, 'isthing': 0, 'name': 'outlier'},
    {'color': (159, 208, 250), 'id': 57, 'isthing': 1, 'name': 'heavy_machinery'},
    {'color': (154, 138, 255), 'id': 58, 'isthing': 1, 'name': 'container'},
    {'color': (23, 211, 232), 'id': 59, 'isthing': 0, 'name': 'hedge'},
    {'color': (0, 208, 208), 'id': 60, 'isthing': 1, 'name': 'barrel'},
    {'color': (0, 0, 221), 'id': 61, 'isthing': 0, 'name': 'pipe'},
    {'color': (132, 164, 196), 'id': 62, 'isthing': 0, 'name': 'tree_root'},
    {'color': (64, 64, 64), 'id': 63, 'isthing': 1, 'name': 'military_vehicle'},
]

# 小区域过滤阈值，按需调整
MIN_SEG_AREA = 1


# ============================================================
# 工具函数
# ============================================================

def id2rgb(segment_id: int):
    """COCO panoptic: segment_id -> RGB"""
    r = segment_id % 256
    g = (segment_id // 256) % 256
    b = (segment_id // 65536) % 256
    return np.array([r, g, b], dtype=np.uint8)


def rgb2id(color: np.ndarray):
    """RGB -> segment_id map"""
    return (
        color[:, :, 0].astype(np.int32)
        + 256 * color[:, :, 1].astype(np.int32)
        + 256 * 256 * color[:, :, 2].astype(np.int32)
    )


def compute_bbox(mask: np.ndarray):
    """mask -> [x, y, w, h]"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def load_mask(path: Path):
    """读取单通道 mask，保持原始整数值"""
    return np.array(Image.open(path))


def connected_components(mask: np.ndarray):
    """
    将二值 mask 拆成连通域。
    返回：每个连通域的 bool mask 列表。
    """
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    comps = []
    for label_id in range(1, num_labels):
        comp = labels == label_id
        if comp.sum() > 0:
            comps.append(comp)
    return comps


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_PANOPTIC_DIR, exist_ok=True)

    cat_map = {c["id"]: c for c in CATEGORIES}

    # 按原图文件名建立索引，便于自动匹配
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(Path(IMAGE_DIR).glob(ext))
    image_files = sorted(image_files)

    semantic_map = {}
    for p in Path(SEMANTIC_DIR).glob("*.png"):
        semantic_map[p.stem] = p

    instance_map = {}
    for p in Path(INSTANCE_DIR).glob("*.png"):
        instance_map[p.stem] = p

    images = []
    annotations = []
    segment_global_id = 1

    for image_id, img_path in enumerate(tqdm(image_files, desc="Building panoptic"), start=1):
        stem = img_path.stem

        sem_path = semantic_map.get(stem)
        ins_path = instance_map.get(stem)

        if sem_path is None:
            print(f"[skip] missing semantic mask for: {img_path.name}")
            continue
        if ins_path is None:
            print(f"[skip] missing instance mask for: {img_path.name}")
            continue

        img = Image.open(img_path)
        w, h = img.size

        semantic = load_mask(sem_path)
        instance = load_mask(ins_path)

        if semantic.shape != instance.shape:
            raise ValueError(
                f"shape mismatch: {img_path.name}, "
                f"semantic={semantic.shape}, instance={instance.shape}"
            )

        # panoptic id map: 每个 segment 一个全局 id
        panoptic_id_map = np.zeros((h, w), dtype=np.int32)
        panoptic_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        segments_info = []

        unique_semantic_ids = np.unique(semantic)

        for class_id in unique_semantic_ids:
            class_id = int(class_id)

            if class_id not in cat_map:
                continue

            cat = cat_map[class_id]
            class_mask = semantic == class_id

            if class_mask.sum() == 0:
                continue

            # ------------------------------------------------
            # thing: 按 instance id 拆
            # ------------------------------------------------
            if cat["isthing"] == 1:
                instance_ids = np.unique(instance[class_mask])
                for inst_id in instance_ids:
                    inst_id = int(inst_id)
                    if inst_id == 0:
                        continue

                    mask = class_mask & (instance == inst_id)
                    area = int(mask.sum())
                    if area < MIN_SEG_AREA:
                        continue

                    # 如果同一实例被切成多个不连通块，可以继续拆分
                    comps = connected_components(mask)
                    if len(comps) == 0:
                        continue

                    for comp in comps:
                        comp_area = int(comp.sum())
                        if comp_area < MIN_SEG_AREA:
                            continue

                        seg_id = segment_global_id
                        segment_global_id += 1

                        panoptic_id_map[comp] = seg_id
                        panoptic_rgb[comp] = id2rgb(seg_id)

                        segments_info.append({
                            "id": int(seg_id),
                            "category_id": int(class_id),
                            "area": int(comp_area),
                            "bbox": compute_bbox(comp),
                            "iscrowd": 0
                        })

            # ------------------------------------------------
            # stuff: 按连通域拆
            # ------------------------------------------------
            else:
                comps = connected_components(class_mask)
                for comp in comps:
                    comp_area = int(comp.sum())
                    if comp_area < MIN_SEG_AREA:
                        continue

                    seg_id = segment_global_id
                    segment_global_id += 1

                    panoptic_id_map[comp] = seg_id
                    panoptic_rgb[comp] = id2rgb(seg_id)

                    segments_info.append({
                        "id": int(seg_id),
                        "category_id": int(class_id),
                        "area": int(comp_area),
                        "bbox": compute_bbox(comp),
                        "iscrowd": 0
                    })

        # 保存 panoptic png，文件名与原图同 stem，后缀固定 png
        panoptic_file_name = f"{stem}.png"
        out_png_path = Path(OUTPUT_PANOPTIC_DIR) / panoptic_file_name
        Image.fromarray(panoptic_rgb).save(out_png_path)

        # images: 保持 COCO 风格
        images.append({
            "id": int(image_id),
            "width": int(w),
            "height": int(h),
            "file_name": img_path.name
        })

        # annotations: 与你第一份 panoptic json 风格一致
        annotations.append({
            "file_name": panoptic_file_name,
            "image_id": int(image_id),
            "segments_info": segments_info
        })

    panoptic_json = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    os.makedirs(Path(OUTPUT_JSON).parent, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(panoptic_json, f, ensure_ascii=False, indent=2)

    print("===================================")
    print(f"saved json: {OUTPUT_JSON}")
    print(f"panoptic png dir: {OUTPUT_PANOPTIC_DIR}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(annotations)}")
    print("===================================")


if __name__ == "__main__":
    main()