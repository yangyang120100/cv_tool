#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import json
from pathlib import Path
import argparse
import random

def load_classes(classes_path):
    if not classes_path:
        return None
    p = Path(classes_path)
    if not p.exists():
        print(f"[WARN] classes file not found: {classes_path}")
        return None
    with p.open('r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def random_color(seed):
    random.seed(seed)
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    # 输入：归一化 x_center,y_center,w,h -> 返回像素坐标 xmin,ymin,xmax,ymax（整数）
    cx = float(xc) * img_w
    cy = float(yc) * img_h
    bw = float(w) * img_w
    bh = float(h) * img_h
    xmin = int(round(cx - bw / 2.0))
    ymin = int(round(cy - bh / 2.0))
    xmax = int(round(cx + bw / 2.0))
    ymax = int(round(cy + bh / 2.0))
    # clip
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w - 1, xmax)
    ymax = min(img_h - 1, ymax)
    return xmin, ymin, xmax, ymax


def draw_labels_on_image(img_path, label_path, classes=None,
                         color_map=None, thickness=2, font_scale=0.6):
    """
    支持 YOLO txt 和 LabelMe JSON 绘制
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]

    bboxes = []

    if label_path.suffix.lower() == '.txt' and label_path.exists():
        # 原 YOLO txt
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls_id = parts[0]
            xc, yc, bw, bh = map(float, parts[1:5])
            cx = xc * w
            cy = yc * h
            bw *= w
            bh *= h
            xmin = max(0, int(round(cx - bw/2)))
            ymin = max(0, int(round(cy - bh/2)))
            xmax = min(w-1, int(round(cx + bw/2)))
            ymax = min(h-1, int(round(cy + bh/2)))
            bboxes.append((cls_id, xmin, ymin, xmax, ymax))

    elif label_path.suffix.lower() == '.json' and label_path.exists():
        # LabelMe JSON
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            pts = shape.get("points", [])
            if len(pts) == 0:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin = max(0, int(min(xs)))
            ymin = max(0, int(min(ys)))
            xmax = min(w-1, int(max(xs)))
            ymax = min(h-1, int(max(ys)))
            cls_id = shape.get("label", "0")
            bboxes.append((cls_id, xmin, ymin, xmax, ymax))

    count = 0
    for cls_id, xmin, ymin, xmax, ymax in bboxes:
        try:
            cls_int = int(cls_id)
        except:
            cls_int = hash(cls_id) & 0xFFFF

        color = random_color(cls_int) if color_map is None else color_map.get(cls_int, random_color(cls_int))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        label_text = str(cls_id)
        if classes and 0 <= int(cls_id) < len(classes):
            label_text = classes[int(cls_id)]

        # 文本背景
        ((text_w, text_h), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = xmin
        text_y = ymin - 5 if ymin - 5 > text_h else ymin + text_h + 5
        cv2.rectangle(img, (text_x, text_y - text_h - 2), (text_x + text_w, text_y + 2), color, -1)
        cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
        count += 1

    return img, count

def build_color_map(num_classes, seed=0):
    cmap = {}
    for i in range(num_classes):
        cmap[i] = random_color(seed + i)
    return cmap

def process_folder(images_dir, labels_dir, out_dir, classes_path=None,
                   save_undetected=False, ext_list=None, num_workers=1):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_path) if classes_path else None

    # create extension list
    if ext_list is None:
        ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # list images
    img_files = [p for p in images_dir.rglob('*') if p.suffix.lower() in ext_list and p.is_file()]
    total = len(img_files)
    print(f"[INFO] Found {total} images in {images_dir}")

    # optional color map
    num_cls = len(classes) if classes else 0
    color_map = build_color_map(num_cls) if num_cls > 0 else None

    processed = 0
    total_labels = 0
    for i, img_path in enumerate(img_files, 1):
        # label file: same basename with .txt under labels_dir, or in same folder (try both)
        rel = img_path.relative_to(images_dir)
        label_path1 = labels_dir / rel.with_suffix('.txt')
        label_path2 = img_path.with_suffix('.txt')  # fallback

        if label_path1.exists():
            label_path = label_path1
        elif label_path2.exists():
            label_path = label_path2
        else:
            label_path = label_path1  # will be absent

        try:
            out_img, cnt = draw_labels_on_image(img_path, label_path, classes=classes, color_map=color_map)
            cv2.imshow('draw_img',cv2.resize(out_img,(1080,640)))
            cv2.waitKey(0)
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            continue

        # save only if has labels or user wants save_undetected
        if cnt > 0 or save_undetected:
            save_to = out_dir / rel
            save_to.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.jpg', out_img)[1].tofile(str(save_to))  # better for unicode path on windows
        processed += 1
        total_labels += cnt
        if i % 200 == 0 or i == total:
            print(f"[INFO] Processed {i}/{total} images. Labeled boxes total: {total_labels}")

    print(f"[DONE] Processed {processed} images, total boxes drawn: {total_labels}")

def process_single_image(image_path, labels_dir, out_path, classes_path=None, save_undetected=True):
    classes = load_classes(classes_path) if classes_path else None
    img_path = Path(image_path)
    labels_dir = Path(labels_dir)
    label_path1 = labels_dir / img_path.with_suffix('.txt').name
    label_path2 = img_path.with_suffix('.txt')
    label_path = label_path1 if label_path1.exists() else label_path2
    out_img, cnt = draw_labels_on_image(img_path, label_path, classes=classes)
    # out_path = Path(out_path)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # if cnt > 0 or save_undetected:
    #     cv2.imencode('.jpg', out_img)[1].tofile(str(out_path))
    cv2.imshow('draw_img',cv2.resize(out_img,(1080,640)))
    cv2.waitKey(0)
    print(f"[DONE] saved {out_path} (drawn boxes: {cnt})")

def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLO labels on images")
    parser.add_argument('--images',default=r"D:\DataBase\Person detection.v16i.yolov8\train\images", help='images folder or single image path')
    parser.add_argument('--labels',default=r"D:\DataBase\Person detection.v16i.yolov8\train\labels", help='labels folder (matching image basenames) or label file path')
    parser.add_argument('--out', default=r"D:\Projects\Scripting tool\drawer\draw", help='output folder or single output image path')
    parser.add_argument('--classes', default=r"D:\DataBase\sdgd_datas\classes.txt", help='optional classes.txt file (one class per line)')
    parser.add_argument('--save_undetected', action='store_true', help='also save images without labels')
    parser.add_argument('--ext', nargs='*', default=['.jpg','.jpeg','.png','.bmp','.tif','.tiff'], help='image extensions to process')
    return parser.parse_args()


#将标签绘制在图像上进行显示
if __name__ == '__main__':
    args = parse_args()
    images = Path(args.images)
    labels = Path(args.labels)
    out = Path(args.out)

    if images.is_file():
        # single image mode
        process_single_image(images, labels, out, classes_path=args.classes, save_undetected=args.save_undetected)
    else:
        process_folder(images, labels, out, classes_path=args.classes,
                       save_undetected=False, ext_list=[e.lower() for e in args.ext])
