import cv2
import json
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

ALL_CLASSES =['background', 'asphalt-road', 'concrete-pavement', 'brick-road', 'dirt', 'gravel', 'inland-water', 'standing-water', 'woods', 'tall-vegetation', 'low-vegetation', 'roof', 'wall', 'fence', 'fence-post', 'snow', 'concrete-pole', 'angle-steel-tower', 'steel-tube-tower', 'conductor', 'sky', 'obstacle', 'car', 'bus', 'work-vehicle', 'large-vehicle', 'bicycle', 'person', 'manhole-cover', 'distant-building', 'traffic-light', 'house', 'tricycle']
# ALL_CLASSES =[
#     "background",      # 0
#     "hard pavement",   # 1
#     "soil",            # 2
#     "gravel",          # 3
#     "inland water",    # 4
#     "standing water",  # 5
#     "vegetation",      # 6
#     "building",        # 7
#     "pole",      # 8
#     "sky",             # 9
#     "obstacle",        # 10
#     "vehicle",         # 11
#     "person",          # 12
#     "manhole cover"    # 13
# ]
# ALL_CLASSES = ['undefined', 'traffic_cone', 'snow', 'cobble', 'obstacle', 'leaves', 'street_light', 'bikeway',
#                'ego_vehicle', 'pedestrian_crossing', 'road_block', 'road_marking', 'car', 'bicycle', 'person', 'bus',
#                'forest', 'bush', 'moss', 'traffic_light', 'motorcycle', 'sidewalk', 'curb', 'asphalt', 'gravel',
#                'boom_barrier', 'rail_track', 'tree_crown', 'tree_trunk', 'debris', 'crops', 'soil', 'rider', 'animal',
#                'truck', 'on_rails', 'caravan', 'trailer', 'building', 'wall', 'rock', 'fence', 'guard_rail', 'bridge',
#                'tunnel', 'pole', 'traffic_sign', 'misc_sign', 'barrier_tape', 'kick_scooter', 'low_grass', 'high_grass',
#                'scenery_vegetation', 'sky', 'water', 'wire', 'outlier', 'heavy_machinery', 'container', 'hedge',
#                'barrel', 'pipe', 'tree_root', 'military_vehicle']

CLASS_MAP = {i: j for i, j in enumerate(ALL_CLASSES)}

IMAGE_SUFFIX = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def mask_to_labelme(
        mask_path,
        image_path,
        output_json,
        min_area=50,
        epsilon_ratio=0.003
):
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(
            f"cannot read mask: {mask_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"cannot read image: {image_path}"
        )

    h, w = image.shape[:2]

    try:
        image_rel_path = os.path.relpath(
            image_path,
            start=output_json.parent
        ).replace("\\", "/")
    except ValueError:
        # 跨盘符时无法生成相对路径，使用绝对路径（或仅文件名）
        image_rel_path = str(image_path).replace("\\", "/")

    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_rel_path,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w)
    }

    class_ids = np.unique(mask)

    for cls_id in class_ids:

        if cls_id == 0:
            continue

        binary = (
            mask == cls_id
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < min_area:
                continue

            epsilon = (
                epsilon_ratio *
                cv2.arcLength(cnt, True)
            )

            poly = cv2.approxPolyDP(
                cnt,
                epsilon,
                True
            )

            if len(poly) < 3:
                continue

            points = (
                poly
                .reshape(-1, 2)
                .astype(float)
                .tolist()
            )

            labelme_data["shapes"].append({
                "label": CLASS_MAP.get(
                    int(cls_id),
                    str(int(cls_id))
                ),
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            })

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
            output_json,
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(
            labelme_data,
            f,
            ensure_ascii=False,
            indent=2
        )

def batch_convert(mask_dir, image_dir, json_dir, min_area=50, epsilon_ratio=0.003, num_workers=1):
    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    image_map = {}
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in IMAGE_SUFFIX:
            image_map[img_path.stem] = img_path

    mask_files = sorted(mask_dir.glob("*.png"))
    print(f"find mask: {len(mask_files)}")

    # 准备任务
    tasks = []
    skipped = 0
    for mask_path in mask_files:
        stem = mask_path.stem
        image_path = image_map.get(stem)
        if image_path is None:
            print(f"[skip] image not found: {stem}")
            skipped += 1
            continue
        output_json = json_dir / f"{stem}.json"
        tasks.append((mask_path, image_path, output_json))

    if num_workers <= 1:
        # 单线程 + 进度条
        success = 0
        failed = 0
        with tqdm(tasks, desc="Converting", unit="file") as pbar:
            for mask_path, image_path, output_json in pbar:
                stem = mask_path.stem
                try:
                    mask_to_labelme(mask_path, image_path, output_json, min_area, epsilon_ratio)
                    success += 1
                except Exception as e:
                    failed += 1
                    tqdm.write(f"[fail] {stem}: {e}")
                pbar.set_postfix(success=success, failed=failed)
    else:
        # 多线程 + 进度条
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        lock = threading.Lock()
        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_stem = {}
            for mask_path, image_path, output_json in tasks:
                stem = mask_path.stem
                future = executor.submit(mask_to_labelme, mask_path, image_path, output_json, min_area, epsilon_ratio)
                future_to_stem[future] = stem

            with tqdm(total=len(tasks), desc="Converting", unit="file") as pbar:
                for future in as_completed(future_to_stem):
                    stem = future_to_stem[future]
                    try:
                        future.result()
                        with lock:
                            success += 1
                    except Exception as e:
                        with lock:
                            failed += 1
                        tqdm.write(f"[fail] {stem}: {e}")
                    with lock:
                        pbar.update(1)
                        pbar.set_postfix(success=success, failed=failed)

    print("\n==========")
    print(f"success : {success}")
    print(f"failed  : {failed}")
    print(f"skipped : {skipped}")
    print("==========")


if __name__ == "__main__":
    batch_convert(
        mask_dir=r"E:\test_segment\remasks",
        image_dir=r"E:\test_segment\images",
        json_dir=r"E:\test_segment\re_jsons",
        min_area=4096,
        epsilon_ratio=0.001,
        num_workers=16
    )
