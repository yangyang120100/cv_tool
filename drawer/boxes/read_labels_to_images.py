#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 YOLO txt 或 LabelMe JSON 标注绘制到图像上
"""
import os
import json
from pathlib import Path
import argparse
import colorsys
import zlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

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

def generate_yolo_colors(n_classes, saturation=1.0, value=1.0, hue_offset=0.0):
    """
    生成类似 YOLO 风格的 RGB 颜色列表，索引 0 固定为红色。

    Args:
        n_classes: 类别总数
        saturation: 饱和度 (0~1)，默认 1.0 使颜色鲜艳
        value: 明度 (0~1)，默认 1.0
        hue_offset: 色相偏移量，默认 0.0（从红色开始）

    Returns:
        list of tuple: 长度为 n_classes 的 RGB 颜色列表 (0~255)
    """
    colors = []
    for i in range(n_classes):
        if i == 0:
            colors.append((255, 0, 0))
        else:
            hue = ((i - 1) / (n_classes - 1) + hue_offset) % 1.0 if n_classes > 1 else 0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    cx = float(xc) * img_w
    cy = float(yc) * img_h
    bw = float(w) * img_w
    bh = float(h) * img_h
    xmin = int(round(cx - bw / 2.0))
    ymin = int(round(cy - bh / 2.0))
    xmax = int(round(cx + bw / 2.0))
    ymax = int(round(cy + bh / 2.0))
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w - 1, xmax)
    ymax = min(img_h - 1, ymax)
    return xmin, ymin, xmax, ymax

def draw_labels_on_image(
        img_path,
        label_path,
        classes=None,
        color_map=None,
        thickness=2,
        font_scale=0.9,
        mask_alpha=0.5
):
    """
    支持：
    - YOLO txt：绘制 bbox
    - LabelMe JSON：绘制 polygon mask（半透明）
    """
    if color_map is None:
        default_colors = generate_yolo_colors(80)
        color_map = {i: default_colors[i] for i in range(80)}

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    overlay = img.copy()
    count = 0

    # ================= YOLO TXT =================
    if label_path and label_path.suffix.lower() == '.txt' and label_path.exists():
        with open(label_path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls_id = parts[0]
                xc, yc, bw, bh = map(float, parts[1:5])
                xmin, ymin, xmax, ymax = yolo_to_bbox(xc, yc, bw, bh, w, h)

                try:
                    cls_int = int(cls_id)
                except:
                    cls_int = hash(cls_id) & 0xFFFF

                color = color_map[cls_int % len(color_map)]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

                label_text = classes[int(cls_id)] if classes and cls_id.isdigit() else str(cls_id)
                cv2.putText(img, label_text, (xmin, max(0, ymin - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                count += 1

    # ================= LabelMe JSON =================
    elif label_path and label_path.suffix.lower() == '.json' and label_path.exists():
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            label = shape.get("label", "unknown")
            points = shape.get("points", [])
            if len(points) < 3:
                continue

            pts = [(int(p[0]), int(p[1])) for p in points]
            pts_np = np.array(pts, dtype=np.int32)

            cls_int = zlib.crc32(label.encode()) & 0xFFFFFFFF
            color = color_map[cls_int % len(color_map)]

            cv2.fillPoly(overlay, [pts_np], color)
            cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=thickness)
            x0, y0 = pts_np[0]
            cv2.putText(img, label, (x0, max(0, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            count += 1

        img = cv2.addWeighted(overlay, mask_alpha, img, 1 - mask_alpha, 0)

    return img, count

def build_color_map(num_classes):
    colors = generate_yolo_colors(num_classes)
    return {i: colors[i] for i in range(num_classes)}

def process_folder(images_dir, labels_dir, out_dir, classes_path=None,
                   save_undetected=False, ext_list=None, num_workers=1):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_path) if classes_path else None

    if ext_list is None:
        ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    img_files = [p for p in images_dir.rglob('*') if p.suffix.lower() in ext_list and p.is_file()]
    total = len(img_files)
    print(f"[INFO] Found {total} images in {images_dir}")

    num_cls = len(classes) if classes else 80
    color_map = build_color_map(num_cls)

    processed = 0
    total_labels = 0
    lock = threading.Lock()

    def process_one(img_path):
        rel = img_path.relative_to(images_dir)
        label_path_txt = labels_dir / rel.with_suffix('.txt')
        label_path_json = labels_dir / rel.with_suffix('.json')

        if label_path_txt.exists():
            label_path = label_path_txt
        elif label_path_json.exists():
            label_path = label_path_json
        else:
            label_path = None

        out_img, cnt = draw_labels_on_image(img_path, label_path, classes=classes, color_map=color_map)
        if cnt > 0 or save_undetected:
            save_to = out_dir / rel
            save_to.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.jpg', out_img)[1].tofile(str(save_to))
        return cnt, rel.stem

    if num_workers <= 1:
        for i, img_path in enumerate(img_files, 1):
            try:
                cnt, stem = process_one(img_path)
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")
                continue
            processed += 1
            total_labels += cnt
            if i % 200 == 0 or i == total:
                print(f"[INFO] Processed {i}/{total} images. Labeled boxes total: {total_labels}")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {executor.submit(process_one, p): p for p in img_files}
            for i, future in enumerate(as_completed(future_to_path), 1):
                img_path = future_to_path[future]
                try:
                    cnt, stem = future.result()
                except Exception as e:
                    with lock:
                        processed += 1
                        print(f"[ERROR] {img_path}: {e}")
                    continue
                with lock:
                    processed += 1
                    total_labels += cnt
                    if processed % 200 == 0 or processed == total:
                        print(f"[INFO] Processed {processed}/{total} images. Labeled boxes total: {total_labels}")

    print(f"[DONE] Processed {processed} images, total boxes drawn: {total_labels}")

def process_single_image(image_path, labels_dir, out_path, classes_path=None, show=False):
    classes = load_classes(classes_path) if classes_path else None
    img_path = Path(image_path)
    labels_dir = Path(labels_dir)
    label_path1 = labels_dir / img_path.with_suffix('.txt').name
    label_path2 = img_path.with_suffix('.txt')
    label_path = label_path1 if label_path1.exists() else label_path2

    num_cls = len(classes) if classes else 80
    color_map = build_color_map(num_cls)

    out_img, cnt = draw_labels_on_image(img_path, label_path, classes=classes, color_map=color_map)
    if show:
        cv2.imshow('draw_img', cv2.resize(out_img, (1080, 1920)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"[DONE] shown image with {cnt} drawn boxes" if show else f"[DONE] processed image with {cnt} boxes")

def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLO labels on images")
    parser.add_argument('--images', default=r"E:\test_images\test_1\images", help='images folder or single image path')
    parser.add_argument('--labels', default=r"E:\test_images\test_1\jsons_2", help='labels folder (matching image basenames) or label file path')
    parser.add_argument('--out', default=r"E:\test_images\test_1\outputs_2", help='output folder or single output image path')
    parser.add_argument('--classes', default=r"E:\test\classes.txt", help='optional classes.txt file (one class per line)')
    parser.add_argument('--save_undetected', action='store_true', help='also save images without labels')
    parser.add_argument('--ext', nargs='*', default=['.jpg','.jpeg','.png','.bmp','.tif','.tiff','.JPG'], help='image extensions to process')
    parser.add_argument('--show', default=False,action='store_true', help='show image in window (only for single image mode)')
    parser.add_argument('--workers', type=int, default=32, help='number of threads for folder processing (default 1)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    images = Path(args.images)
    labels = Path(args.labels)
    out = Path(args.out)

    if images.is_file():
        process_single_image(images, labels, out, classes_path=args.classes, show=args.show)
    else:
        process_folder(images, labels, out, classes_path=args.classes,
                       save_undetected=args.save_undetected,
                       ext_list=[e.lower() for e in args.ext],
                       num_workers=args.workers)