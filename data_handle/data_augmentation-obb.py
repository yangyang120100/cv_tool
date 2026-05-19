# 工业级 OBB 数据增强脚本（优化版）

## 主要改进
import os
import cv2
import json
import copy
import hashlib
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A

# ==========================================
# 可选：polygon 合法性检查
# ==========================================
try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except:
    HAS_SHAPELY = False


# ==========================================
# 全局参数
# ==========================================
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

MIN_AREA = 16
MIN_EDGE = 3
MAX_RATIO = 20
MIN_INSIDE_RATIO = 0.7


# ==========================================
# 安全增强策略（工业推荐）
# ==========================================
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.Affine(
            translate_percent=(-0.03, 0.03),
            scale=(0.95, 1.08),
            rotate=(-12, 12),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.7,
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.GaussNoise(std_range=(0.01, 0.03), p=1),
        ], p=0.15),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.2,
        ),
    ],

    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False,
    )
)


# ==========================================
# Hash
# ==========================================
def fast_hash(img):
    h, w = img.shape[:2]

    sample = img[
        h // 4:h // 4 + 32,
        w // 4:w // 4 + 32
    ]

    return hashlib.md5(sample.tobytes()).hexdigest()


# ==========================================
# Polygon 工具
# ==========================================
def polygon_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    )


# ==========================================
# 点排序（顺时针）
# ==========================================
def order_points_clockwise(pts):
    center = np.mean(pts, axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    pts = pts[np.argsort(angles)]

    start = np.argmin(pts.sum(axis=1))

    return np.roll(pts, -start, axis=0)


# ==========================================
# self intersection check
# ==========================================
def polygon_valid(pts):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) != 4:
        return False

    area = polygon_area(pts)

    if area < MIN_AREA:
        return False

    hull = cv2.convexHull(pts.astype(np.float32))

    if len(hull) < 4:
        return False

    if not cv2.isContourConvex(hull):
        return False

    if HAS_SHAPELY:
        try:
            poly = Polygon(pts)

            if not poly.is_valid:
                return False

            if poly.area < MIN_AREA:
                return False

        except:
            return False

    return True


# ==========================================
# longest edge angle
# ==========================================
def compute_direction(box):
    edges = []

    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = np.sqrt(dx * dx + dy * dy)

        edges.append((length, dx, dy))

    longest = max(edges, key=lambda x: x[0])

    angle = np.arctan2(longest[2], longest[1])

    return float(angle)


# ==========================================
# canonicalize
# 保证 w >= h
# angle in [-90,0)
# ==========================================
def canonicalize_box(box):
    rect = cv2.minAreaRect(box.astype(np.float32))

    (cx, cy), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90

    while angle >= 0:
        angle -= 90

    while angle < -90:
        angle += 90

    rect = ((cx, cy), (w, h), angle)

    pts = cv2.boxPoints(rect)

    pts = order_points_clockwise(pts)

    return pts


# ==========================================
# inside ratio
# ==========================================
def compute_inside_ratio(pts, w, h):
    inside = 0

    for p in pts:
        if 0 <= p[0] < w and 0 <= p[1] < h:
            inside += 1

    return inside / 4.0


# ==========================================
# aspect ratio
# ==========================================
def check_aspect_ratio(box):
    rect = cv2.minAreaRect(box.astype(np.float32))

    (_, _), (w, h), _ = rect

    if w < 1 or h < 1:
        return False

    ratio = max(w, h) / min(w, h)

    if ratio > MAX_RATIO:
        return False

    if min(w, h) < MIN_EDGE:
        return False

    return True


# ==========================================
# rebuild obb
# ==========================================
def rebuild_obb(pts, img_w, img_h):
    pts = np.asarray(pts, dtype=np.float32)

    if not polygon_valid(pts):
        return None, "invalid_polygon"

    inside_ratio = compute_inside_ratio(pts, img_w, img_h)

    if inside_ratio < MIN_INSIDE_RATIO:
        return None, "outside"

    rect = cv2.minAreaRect(pts.astype(np.float32))

    box = cv2.boxPoints(rect)

    box = order_points_clockwise(box)

    box = canonicalize_box(box)

    if not polygon_valid(box):
        return None, "bad_obb"

    if not check_aspect_ratio(box):
        return None, "bad_ratio"

    return box, None


# ==========================================
# TXT
# ==========================================
def load_txt(path):
    objs = []

    if not os.path.exists(path):
        return objs

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 9:
                continue

            pts = np.array(
                list(map(float, parts[1:9]))
            ).reshape(4, 2)

            objs.append({
                "cls": parts[0],
                "pts": pts,
            })

    return objs


# ==========================================
# SAVE TXT
# ==========================================
def save_txt(path, objs):
    with open(path, "w") as f:
        for obj in objs:
            pts = np.round(obj["pts"]).astype(int).reshape(-1)

            line = obj["cls"] + " " + " ".join(map(str, pts))

            f.write(line + "\n")


# ==========================================
# JSON
# ==========================================
def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# extract
# ==========================================
def extract_objects(data):
    objs = []
    indices = []

    for idx, shape in enumerate(data.get("shapes", [])):
        pts = shape.get("points", [])

        if len(pts) != 4:
            continue

        objs.append({
            "cls": shape.get("label", "unknown"),
            "pts": np.array(pts, dtype=np.float32),
        })

        indices.append(idx)

    return objs, indices


# ==========================================
# SAVE JSON
# ==========================================
def save_json(path, template, objs, indices):
    data = copy.deepcopy(template)

    for obj, idx in zip(objs, indices):
        shape = data["shapes"][idx]

        shape["points"] = obj["pts"].tolist()

        shape["direction"] = compute_direction(obj["pts"])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ==========================================
# process one
# ==========================================
def process_one(args_tuple):
    (
        image_name,
        image_dir,
        label_dir,
        out_image_dir,
        out_label_dir,
        label_ext,
        aug_times,
    ) = args_tuple

    fail_stats = {
        "invalid_polygon": 0,
        "outside": 0,
        "bad_obb": 0,
        "bad_ratio": 0,
    }

    stem = Path(image_name).stem

    img_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, stem + label_ext)

    if not os.path.exists(label_path):
        return fail_stats

    image = cv2.imread(img_path)

    if image is None:
        return fail_stats

    # ======================================
    # load labels
    # ======================================
    if label_ext == ".json":
        data = load_json(label_path)

        if data is None:
            return fail_stats

        objs, indices = extract_objects(data)

    else:
        objs = load_txt(label_path)
        data = None
        indices = None

    if len(objs) == 0:
        return fail_stats

    # ======================================
    # save original
    # ======================================
    cv2.imwrite(
        os.path.join(out_image_dir, image_name),
        image,
    )

    if label_ext == ".json":
        save_json(
            os.path.join(out_label_dir, stem + label_ext),
            data,
            objs,
            indices,
        )
    else:
        save_txt(
            os.path.join(out_label_dir, stem + label_ext),
            objs,
        )

    # ======================================
    # augmentation
    # ======================================
    keypoints = []

    for obj in objs:
        keypoints.extend(obj["pts"])

    used_hash = set()
    origin_hash = fast_hash(image)

    for aug_idx in range(aug_times):

        success = False

        for _ in range(15):
            aug = transform(
                image=image,
                keypoints=keypoints,
            )

            aug_img = aug["image"]
            aug_pts = aug["keypoints"]

            if len(aug_pts) != len(keypoints):
                continue

            new_objs = []

            valid = True

            for idx in range(len(objs)):
                pts = np.array(
                    aug_pts[idx * 4:(idx + 1) * 4],
                    dtype=np.float32,
                )

                obb, reason = rebuild_obb(
                    pts,
                    aug_img.shape[1],
                    aug_img.shape[0],
                )

                if obb is None:
                    fail_stats[reason] += 1
                    valid = False
                    break

                new_objs.append({
                    "cls": objs[idx]["cls"],
                    "pts": obb,
                })

            if not valid:
                continue

            hsh = fast_hash(aug_img)

            if hsh == origin_hash:
                continue

            if hsh in used_hash:
                continue

            used_hash.add(hsh)

            new_name = f"{stem}_aug{aug_idx}.jpg"

            cv2.imwrite(
                os.path.join(out_image_dir, new_name),
                aug_img,
            )

            if label_ext == ".json":
                save_json(
                    os.path.join(
                        out_label_dir,
                        Path(new_name).stem + label_ext,
                    ),
                    data,
                    new_objs,
                    indices,
                )
            else:
                save_txt(
                    os.path.join(
                        out_label_dir,
                        Path(new_name).stem + label_ext,
                    ),
                    new_objs,
                )

            success = True
            break

        if not success:
            print(f"[WARN] augmentation failed: {image_name}")

    return fail_stats


# ==========================================
# args
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default=r"D:\DataBase\Insulator_datas\images"
    )

    parser.add_argument(
        "--label_dir",
        type=str,
        default=r"D:\DataBase\Insulator_datas\jsons"
    )

    parser.add_argument(
        "--label_ext",
        type=str,
        default=".json",
    )

    parser.add_argument(
        "--aug_times",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    return parser.parse_args()


# ==========================================
# main
# ==========================================
if __name__ == "__main__":

    args = parse_args()

    image_dir = args.image_dir
    label_dir = args.label_dir
    label_ext = args.label_ext.lower()

    out_image_dir = os.path.join(
        os.path.dirname(image_dir),
        "aug_images"
    )

    out_label_dir = os.path.join(
        os.path.dirname(image_dir),
        "aug_labels"
    )

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    image_names = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in VALID_EXT
    ]

    print("=" * 60)
    print("工业级 OBB 数据增强")
    print("图片数量:", len(image_names))
    print("标签格式:", label_ext)
    print("=" * 60)

    task_args = []

    for name in image_names:
        task_args.append((
            name,
            image_dir,
            label_dir,
            out_image_dir,
            out_label_dir,
            label_ext,
            args.aug_times,
        ))

    all_stats = {
        "invalid_polygon": 0,
        "outside": 0,
        "bad_obb": 0,
        "bad_ratio": 0,
    }

    with ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:

        results = list(
            tqdm(
                executor.map(process_one, task_args),
                total=len(task_args)
            )
        )

    for stats in results:
        for k in all_stats:
            all_stats[k] += stats[k]

    print("\n增强完成")

    print("\n失败统计:")

    for k, v in all_stats.items():
        print(f"{k}: {v}")

