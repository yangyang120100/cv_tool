import os
import cv2
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import albumentations as A

"""
工业级 OBB 数据增强脚本
支持：
- txt (DOTA 四点)
- json (LabelMe polygon)
- 显式指定标签格式
"""

# =====================================================
# 全局变量
# =====================================================
AUG_TIMES = None
NUM_WORKERS = None
IMAGE_DIR = None
LABEL_DIR = None
OUT_IMAGE_DIR = None
OUT_LABEL_DIR = None
LABEL_EXT = None

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# =====================================================
# 增强策略
# =====================================================
transform = A.Compose(
    [
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
        ], p=1.0),

        A.Affine(
            translate_percent=(0.0, 0.08),
            scale=(0.9, 1.15),
            rotate=(-45, 45),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.7
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.GaussNoise(p=1),
        ], p=0.3),

        A.RandomBrightnessContrast(p=0.4),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False
    )
)

# =====================================================
# 工具函数
# =====================================================
def img_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def poly_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_pts(pts, w, h):
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def order_pts(pts):
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = np.argmin(pts.sum(axis=1))
    return np.roll(pts, -start, axis=0)


def fit_obb(pts, w, h):
    pts = clip_pts(pts.copy(), w, h)

    if poly_area(pts) < 10:
        return None

    rect = cv2.minAreaRect(pts.astype(np.float32))
    box = cv2.boxPoints(rect)

    box = clip_pts(box, w, h)
    box = order_pts(box)

    if poly_area(box) < 10:
        return None

    return box


# =====================================================
# TXT 标签
# =====================================================
def load_txt(label_path):
    objs = []
    if not os.path.exists(label_path):
        return objs

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            cls = parts[0]
            pts = np.array(list(map(float, parts[1:9]))).reshape(4, 2)

            objs.append({"cls": cls, "pts": pts})
    return objs


def save_txt(label_path, objs):
    with open(label_path, "w") as f:
        for obj in objs:
            pts = np.round(obj["pts"]).astype(int).reshape(-1)
            line = obj["cls"] + " " + " ".join(map(str, pts))
            f.write(line + "\n")


# =====================================================
# JSON 标签（LabelMe）
# =====================================================
def load_json(label_path):
    objs = []
    if not os.path.exists(label_path):
        return objs

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for s in data.get("shapes", []):
        pts = s.get("points", [])
        if len(pts) != 4:
            continue

        objs.append({
            "cls": s.get("label", "unknown"),
            "pts": np.array(pts, dtype=np.float32)
        })

    return objs


def save_json(label_path, objs):
    data = {"shapes": []}

    for obj in objs:
        data["shapes"].append({
            "label": obj["cls"],
            "points": obj["pts"].tolist(),
            "shape_type": "polygon"
        })

    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =====================================================
# 统一接口
# =====================================================
def load_labels(path):
    if LABEL_EXT == ".json":
        return load_json(path)
    return load_txt(path)


def save_labels(path, objs):
    if LABEL_EXT == ".json":
        save_json(path, objs)
    else:
        save_txt(path, objs)


# =====================================================
# 单图处理
# =====================================================
def process_one(name):
    stem = Path(name).stem

    img_path = os.path.join(IMAGE_DIR, name)
    label_path = os.path.join(LABEL_DIR, stem + LABEL_EXT)

    if not os.path.exists(label_path):
        return

    image = cv2.imread(img_path)
    if image is None:
        return

    objs = load_labels(label_path)
    if len(objs) == 0:
        return

    h, w = image.shape[:2]

    # 保存原始
    cv2.imwrite(os.path.join(OUT_IMAGE_DIR, name), image)
    save_labels(os.path.join(OUT_LABEL_DIR, stem + LABEL_EXT), objs)

    keypoints = []
    classes = []

    for o in objs:
        keypoints.extend(o["pts"])
        classes.append(o["cls"])

    origin_hash = img_hash(image)
    used_hash = set()

    for i in range(AUG_TIMES):
        for _ in range(10):
            aug = transform(image=image, keypoints=keypoints)

            aug_img = aug["image"]
            aug_kps = aug["keypoints"]

            if len(aug_kps) != len(keypoints):
                continue

            new_objs = []
            valid = True

            for idx, cls in enumerate(classes):
                pts = np.array(aug_kps[idx*4:(idx+1)*4])
                obb = fit_obb(pts, aug_img.shape[1], aug_img.shape[0])

                if obb is None:
                    valid = False
                    break

                new_objs.append({"cls": cls, "pts": obb})

            if not valid:
                continue

            hsh = img_hash(aug_img)
            if hsh == origin_hash or hsh in used_hash:
                continue

            used_hash.add(hsh)

            new_name = f"{stem}_aug{i}.jpg"

            cv2.imwrite(os.path.join(OUT_IMAGE_DIR, new_name), aug_img)

            save_labels(
                os.path.join(OUT_LABEL_DIR, Path(new_name).stem + LABEL_EXT),
                new_objs
            )
            break


# =====================================================
# 参数
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", default=r"D:\DataBase\Insulator_datas\images")
    parser.add_argument("--label_dir", default=r"D:\DataBase\Insulator_datas\jsons")
    parser.add_argument("--label_ext", default=".json")

    parser.add_argument("--aug_times", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=32)

    return parser.parse_args()


# =====================================================
# 主函数
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    AUG_TIMES = args.aug_times
    NUM_WORKERS = args.num_workers
    IMAGE_DIR = args.image_dir
    LABEL_DIR = args.label_dir
    LABEL_EXT = args.label_ext.lower()

    OUT_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "aug_images")
    OUT_LABEL_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "aug_labels")

    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)

    image_names = [
        f for f in os.listdir(IMAGE_DIR)
        if Path(f).suffix.lower() in VALID_EXT
    ]

    print("开始增强...")
    print("图片数:", len(image_names))
    print("标签格式:", LABEL_EXT)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        list(tqdm(ex.map(process_one, image_names), total=len(image_names)))

    print("完成")