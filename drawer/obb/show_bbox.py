import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

"""
OBB 标签可视化脚本
支持 txt / json
"""

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# =====================================================
# 读取标签
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


def load_labels(path, ext):
    if ext == ".json":
        return load_json(path)
    return load_txt(path)


# =====================================================
# 绘制
# =====================================================
def draw_obb(image, objs):
    for obj in objs:
        pts = obj["pts"].astype(int)

        # 画四边形
        cv2.polylines(
            image,
            [pts.reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        # 类别文字
        x, y = pts[0]
        cv2.putText(
            image,
            obj["cls"],
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return image


# =====================================================
# 主逻辑
# =====================================================
def process(image_dir, label_dir, out_dir, label_ext):
    os.makedirs(out_dir, exist_ok=True)

    image_names = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in VALID_EXT
    ]

    for name in tqdm(image_names):
        stem = Path(name).stem

        img_path = os.path.join(image_dir, name)
        label_path = os.path.join(label_dir, stem + label_ext)

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        objs = load_labels(label_path, label_ext)

        img = draw_obb(img, objs)


        cv2.imshow("obb", cv2.resize(img,(1080,1080)))
        cv2.waitKey(0)
        # out_path = os.path.join(out_dir, name)
        # cv2.imwrite(out_path, img)


# =====================================================
# 参数
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir",default=r"D:\DataBase\Fuse_isolating-switch_drop-out-lightning_arrester_lightning-arrester\aug_images")
    parser.add_argument("--label_dir", default=r"D:\DataBase\Fuse_isolating-switch_drop-out-lightning_arrester_lightning-arrester\aug_labels")
    parser.add_argument("--out_dir", default="vis")
    parser.add_argument("--label_ext", default=".json")

    return parser.parse_args()


# =====================================================
# 入口
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    process(
        args.image_dir,
        args.label_dir,
        args.out_dir,
        args.label_ext.lower()
    )

    print("可视化完成")