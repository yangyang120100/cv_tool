#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 YOLO txt 或 LabelMe JSON 标注绘制到图像上。

主要功能：
1. YOLO TXT：
   - 绘制 bounding box
   - 显示类别名称
2. LabelMe JSON：
   - 绘制 polygon mask
   - 半透明填充
   - 绘制 polygon 边界
   - 显示类别名称
3. 类别颜色：
   - 根据 class_id 确定性生成
   - HSV -> BGR
   - 同一个 class_id 永远对应同一个颜色
   - 不使用随机数，因此多次运行结果一致
4. 支持多线程批量处理
"""

import os
import json
import zlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ============================================================
# HSV -> BGR
# ============================================================

def hsv2bgr(h, s, v):
    """
    HSV -> BGR

    参数：
        h: [0, 1)
        s: [0, 1]
        v: [0, 1]

    返回：
        (B, G, R)，范围 [0, 255]
    """

    # 防止 h=1.0 导致 h_i=6
    h = float(h) % 1.0
    s = float(np.clip(s, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))

    h_i = int(h * 6)
    f = h * 6 - h_i

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0.0, 0.0, 0.0

    if h_i == 0:
        r, g, b = v, t, p

    elif h_i == 1:
        r, g, b = q, v, p

    elif h_i == 2:
        r, g, b = p, v, t

    elif h_i == 3:
        r, g, b = p, q, v

    elif h_i == 4:
        r, g, b = t, p, v

    else:  # h_i == 5
        r, g, b = v, p, q

    return (
        int(round(b * 255)),
        int(round(g * 255)),
        int(round(r * 255)),
    )


# ============================================================
# 根据 class_id 生成颜色
# ============================================================

def random_color(
    class_id,
    min_s=0.70,
    max_s=1.00,
    min_v=0.90,
    max_v=1.00,
):
    """
    根据 class_id 确定性生成颜色。

    注意：
        虽然函数名保留为 random_color，
        但实际上不是随机生成。
        同一个 class_id 永远得到相同颜色。

    参数：
        class_id:
            类别 ID

        min_s / max_s:
            HSV 饱和度范围。

        min_v / max_v:
            HSV 明度范围。

    返回：
        BGR tuple
    """

    # 转成 int，避免传入 numpy.int64 等类型产生异常行为
    class_id = int(class_id)

    # --------------------------------------------------------
    # Hue
    # --------------------------------------------------------
    #
    # 使用不同的位移 + 异或产生确定性伪随机值。
    #
    # 这里使用更大的整数范围，
    # 避免简单的 id * 固定步长造成明显颜色周期。
    #

    h_seed = (
        ((class_id << 2) ^ 0x937151) *
        2654435761
    ) & 0xFFFFFFFF

    h_plane = (h_seed % 10000) / 10000.0

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    s_seed = (
        ((class_id << 3) ^ 0x315793) *
        2246822519
    ) & 0xFFFFFFFF

    s_norm = (s_seed % 10000) / 10000.0

    s_plane = (
        min_s +
        s_norm * (max_s - min_s)
    )

    # --------------------------------------------------------
    # Value
    # --------------------------------------------------------

    v_seed = (
        ((class_id << 5) ^ 0xA53C9E) *
        3266489917
    ) & 0xFFFFFFFF

    v_norm = (v_seed % 10000) / 10000.0

    v_plane = (
        min_v +
        v_norm * (max_v - min_v)
    )

    return hsv2bgr(
        h_plane,
        s_plane,
        v_plane,
    )


# ============================================================
# 生成所有类别颜色
# ============================================================

def build_color_map(
    num_classes,
    min_s=0.70,
    max_s=1.00,
    min_v=0.90,
    max_v=1.00,
):
    """
    为 0 ~ num_classes-1 生成颜色。

    返回：
        {
            0: (B, G, R),
            1: (B, G, R),
            ...
        }
    """

    if num_classes <= 0:
        return {}

    color_map = {}

    for class_id in range(num_classes):
        color_map[class_id] = random_color(
            class_id,
            min_s=min_s,
            max_s=max_s,
            min_v=min_v,
            max_v=max_v,
        )

    return color_map


# ============================================================
# 类别读取
# ============================================================

def load_classes(classes_path):
    """
    从 classes.txt 读取类别名称。

    格式：
        class0
        class1
        class2
        ...

    返回：
        list[str]
    """

    if not classes_path:
        return None

    p = Path(classes_path)

    if not p.exists():
        print(f"[WARN] classes file not found: {classes_path}")
        return None

    with p.open("r", encoding="utf-8") as f:
        names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"[INFO] Loaded {len(names)} classes")

    return names


# ============================================================
# 构建 label -> class_id 映射
# ============================================================

def build_label_to_id(classes):
    """
    根据 classes.txt 建立：

        类别名称 -> class_id

    例如：

        Background -> 0
        柏油马路 -> 1
        水泥路面 -> 2
    """

    if not classes:
        return {}

    return {
        str(name): idx
        for idx, name in enumerate(classes)
    }


# ============================================================
# YOLO 坐标转换
# ============================================================

def yolo_to_bbox(
    xc,
    yc,
    w,
    h,
    img_w,
    img_h,
):
    """
    YOLO 归一化坐标 -> 像素 bbox
    """

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

    return (
        xmin,
        ymin,
        xmax,
        ymax,
    )


# ============================================================
# YOLO TXT
# ============================================================

def draw_yolo_labels(
    img,
    label_path,
    classes,
    color_map,
    thickness=2,
    font_scale=0.9,
):
    """
    绘制 YOLO TXT bbox。
    """

    h, w = img.shape[:2]

    count = 0

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            cls_id_str = parts[0]

            try:
                cls_id = int(cls_id_str)
            except ValueError:
                print(
                    f"[WARN] Invalid class id: "
                    f"{cls_id_str}"
                )
                continue

            try:
                xc, yc, bw, bh = map(
                    float,
                    parts[1:5],
                )
            except ValueError:
                print(
                    f"[WARN] Invalid bbox: {line}"
                )
                continue

            if cls_id not in color_map:
                print(
                    f"[WARN] Unknown class id: {cls_id}"
                )
                continue

            xmin, ymin, xmax, ymax = yolo_to_bbox(
                xc,
                yc,
                bw,
                bh,
                w,
                h,
            )

            color = color_map[cls_id]

            # bbox
            cv2.rectangle(
                img,
                (xmin, ymin),
                (xmax, ymax),
                color,
                thickness,
            )

            # 类别名称
            if (
                classes is not None
                and 0 <= cls_id < len(classes)
            ):
                label_text = classes[cls_id]
            else:
                label_text = str(cls_id)

            # 文字背景
            (
                text_w,
                text_h,
            ), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                2,
            )

            text_x = xmin
            text_y = max(
                text_h + baseline + 2,
                ymin - 5,
            )

            cv2.rectangle(
                img,
                (
                    text_x,
                    text_y - text_h - baseline - 2,
                ),
                (
                    text_x + text_w + 4,
                    text_y + 2,
                ),
                color,
                -1,
            )

            cv2.putText(
                img,
                label_text,
                (
                    text_x + 2,
                    text_y - 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            count += 1

    return count


# ============================================================
# LabelMe JSON
# ============================================================

def draw_labelme_labels(
    img,
    label_path,
    classes,
    color_map,
    label_to_id,
    thickness=2,
    font_scale=0.9,
    mask_alpha=0.5,
):
    """
    绘制 LabelMe JSON polygon。
    """

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    overlay = img.copy()

    count = 0

    shapes = data.get(
        "shapes",
        [],
    )

    for shape in shapes:

        label = str(
            shape.get(
                "label",
                "unknown",
            )
        )

        points = shape.get(
            "points",
            [],
        )

        if len(points) < 3:
            continue

        # ----------------------------------------------------
        # 获取 class_id
        # ----------------------------------------------------

        if label in label_to_id:

            cls_id = label_to_id[label]

        else:

            # ------------------------------------------------
            # 如果没有 classes.txt 对应关系，
            # 采用稳定 hash 映射。
            # ------------------------------------------------
            #
            # 注意：
            # 这种方法不能保证不碰撞。
            #

            cls_id = zlib.crc32(
                label.encode("utf-8")
            ) & 0xFFFFFFFF

            if len(color_map) > 0:
                cls_id %= len(color_map)
            else:
                continue

            print(
                f"[WARN] Label '{label}' "
                f"not found in classes.txt, "
                f"use fallback class id={cls_id}"
            )

        if cls_id not in color_map:
            print(
                f"[WARN] Class id {cls_id} "
                f"not found in color map"
            )
            continue

        # ----------------------------------------------------
        # points
        # ----------------------------------------------------

        pts = []

        for p in points:

            if len(p) < 2:
                continue

            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
            except (TypeError, ValueError):
                continue

            pts.append((x, y))

        if len(pts) < 3:
            continue

        pts_np = np.asarray(
            pts,
            dtype=np.int32,
        )

        color = color_map[cls_id]

        # ----------------------------------------------------
        # 填充 mask
        # ----------------------------------------------------

        cv2.fillPoly(
            overlay,
            [pts_np],
            color,
        )

        # ----------------------------------------------------
        # polygon 边界
        # ----------------------------------------------------

        cv2.polylines(
            img,
            [pts_np],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # 标签位置
        # ----------------------------------------------------

        x0, y0 = pts_np[0]

        if (
            classes is not None
            and 0 <= cls_id < len(classes)
        ):
            label_text = classes[cls_id]
        else:
            label_text = label

        (
            text_w,
            text_h,
        ), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2,
        )

        text_x = max(0, x0)

        text_y = max(
            text_h + baseline + 2,
            y0,
        )

        # 文字背景
        cv2.rectangle(
            img,
            (
                text_x,
                text_y - text_h - baseline - 2,
            ),
            (
                text_x + text_w + 4,
                text_y + 2,
            ),
            color,
            -1,
        )

        cv2.putText(
            img,
            label_text,
            (
                text_x + 2,
                text_y - 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        count += 1

    # --------------------------------------------------------
    # mask alpha
    # --------------------------------------------------------

    img[:] = cv2.addWeighted(
        overlay,
        mask_alpha,
        img,
        1.0 - mask_alpha,
        0,
    )

    return count


# ============================================================
# 单张图绘制
# ============================================================

def draw_labels_on_image(
    img_path,
    label_path,
    classes=None,
    color_map=None,
    label_to_id=None,
    thickness=2,
    font_scale=0.9,
    mask_alpha=0.5,
):
    """
    自动判断：
        .txt -> YOLO
        .json -> LabelMe
    """

    # --------------------------------------------------------
    # 默认颜色
    # --------------------------------------------------------

    if color_map is None:

        num_classes = (
            len(classes)
            if classes
            else 80
        )

        color_map = build_color_map(
            num_classes
        )

    if label_to_id is None:
        label_to_id = build_label_to_id(
            classes
        )

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    img = cv2.imread(
        str(img_path),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise RuntimeError(
            f"Cannot read image: {img_path}"
        )

    count = 0

    # --------------------------------------------------------
    # 无标签
    # --------------------------------------------------------

    if label_path is None:
        return img, count

    label_path = Path(label_path)

    if not label_path.exists():
        return img, count

    suffix = label_path.suffix.lower()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    if suffix == ".txt":

        count = draw_yolo_labels(
            img,
            label_path,
            classes,
            color_map,
            thickness=thickness,
            font_scale=font_scale,
        )

    # --------------------------------------------------------
    # LabelMe
    # --------------------------------------------------------

    elif suffix == ".json":

        count = draw_labelme_labels(
            img,
            label_path,
            classes,
            color_map,
            label_to_id,
            thickness=thickness,
            font_scale=font_scale,
            mask_alpha=mask_alpha,
        )

    else:

        print(
            f"[WARN] Unsupported label format: "
            f"{label_path}"
        )

    return img, count


# ============================================================
# 保存图像
# ============================================================

def save_image(
    image,
    save_path,
):
    """
    使用 cv2.imencode + tofile，
    避免 Windows 中文路径问题。
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = save_path.suffix.lower()

    if suffix not in (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    ):
        suffix = ".jpg"

    ok, encoded = cv2.imencode(
        suffix,
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to encode image: "
            f"{save_path}"
        )

    encoded.tofile(
        str(save_path)
    )


# ============================================================
# 批量处理
# ============================================================

def process_folder(
    images_dir,
    labels_dir,
    out_dir,
    classes_path=None,
    save_undetected=False,
    ext_list=None,
    num_workers=1,
):
    """
    批量处理文件夹。
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # classes
    # --------------------------------------------------------

    classes = (
        load_classes(classes_path)
        if classes_path
        else None
    )

    label_to_id = build_label_to_id(
        classes
    )

    # --------------------------------------------------------
    # extensions
    # --------------------------------------------------------

    if ext_list is None:

        ext_list = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ]

    ext_set = {
        e.lower()
        for e in ext_list
    }

    # --------------------------------------------------------
    # images
    # --------------------------------------------------------

    img_files = [
        p
        for p in images_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in ext_set
    ]

    total = len(img_files)

    print(
        f"[INFO] Found {total} images "
        f"in {images_dir}"
    )

    # --------------------------------------------------------
    # color map
    # --------------------------------------------------------

    if classes:

        num_cls = len(classes)

    else:

        # 没有 classes.txt 时使用 80 类
        num_cls = 80

    color_map = build_color_map(
        num_cls,
        min_s=0.70,
        max_s=1.00,
        min_v=0.90,
        max_v=1.00,
    )

    print(
        f"[INFO] Generated "
        f"{len(color_map)} colors"
    )

    # --------------------------------------------------------
    # counters
    # --------------------------------------------------------

    processed = 0
    total_labels = 0

    lock = threading.Lock()

    # ========================================================
    # 单张图片
    # ========================================================

    def process_one(img_path):

        rel = img_path.relative_to(
            images_dir
        )

        label_path_txt = (
            labels_dir /
            rel.with_suffix(".txt")
        )

        label_path_json = (
            labels_dir /
            rel.with_suffix(".json")
        )

        # ----------------------------------------------------
        # 找 label
        # ----------------------------------------------------

        if label_path_txt.exists():

            label_path = label_path_txt

        elif label_path_json.exists():

            label_path = label_path_json

        else:

            label_path = None

        # ----------------------------------------------------
        # 绘制
        # ----------------------------------------------------

        out_img, cnt = draw_labels_on_image(
            img_path,
            label_path,
            classes=classes,
            color_map=color_map,
            label_to_id=label_to_id,
        )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        if cnt > 0 or save_undetected:

            save_to = (
                out_dir / rel
            )

            save_image(
                out_img,
                save_to,
            )

        return cnt, rel.stem

    # ========================================================
    # 单线程
    # ========================================================

    if num_workers <= 1:

        for i, img_path in enumerate(
            img_files,
            1,
        ):

            try:

                cnt, stem = process_one(
                    img_path
                )

            except Exception as e:

                print(
                    f"[ERROR] "
                    f"{img_path}: {e}"
                )

                continue

            processed += 1
            total_labels += cnt

            if (
                i % 200 == 0
                or i == total
            ):

                print(
                    f"[INFO] Processed "
                    f"{i}/{total} images. "
                    f"Labeled objects total: "
                    f"{total_labels}"
                )

    # ========================================================
    # 多线程
    # ========================================================

    else:

        with ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:

            future_to_path = {
                executor.submit(
                    process_one,
                    p,
                ): p
                for p in img_files
            }

            for i, future in enumerate(
                as_completed(
                    future_to_path
                ),
                1,
            ):

                img_path = (
                    future_to_path[future]
                )

                try:

                    cnt, stem = (
                        future.result()
                    )

                except Exception as e:

                    with lock:

                        processed += 1

                        print(
                            f"[ERROR] "
                            f"{img_path}: {e}"
                        )

                    continue

                with lock:

                    processed += 1
                    total_labels += cnt

                    if (
                        processed % 200 == 0
                        or processed == total
                    ):

                        print(
                            f"[INFO] Processed "
                            f"{processed}/{total} "
                            f"images. "
                            f"Labeled objects total: "
                            f"{total_labels}"
                        )

    print(
        f"[DONE] Processed "
        f"{processed} images, "
        f"total objects drawn: "
        f"{total_labels}"
    )


# ============================================================
# 单张图片处理
# ============================================================

def process_single_image(
    image_path,
    labels_dir,
    out_path,
    classes_path=None,
    show=False,
):
    """
    单张图片处理。
    """

    classes = (
        load_classes(classes_path)
        if classes_path
        else None
    )

    label_to_id = build_label_to_id(
        classes
    )

    img_path = Path(
        image_path
    )

    labels_dir = Path(
        labels_dir
    )

    # --------------------------------------------------------
    # txt
    # --------------------------------------------------------

    label_path1 = (
        labels_dir /
        img_path.with_suffix(".txt").name
    )

    # --------------------------------------------------------
    # json
    # --------------------------------------------------------

    label_path2 = (
        labels_dir /
        img_path.with_suffix(".json").name
    )

    # --------------------------------------------------------
    # 同目录可能存在 label
    # --------------------------------------------------------

    label_path3 = (
        img_path.with_suffix(".txt")
    )

    label_path4 = (
        img_path.with_suffix(".json")
    )

    if label_path1.exists():

        label_path = label_path1

    elif label_path2.exists():

        label_path = label_path2

    elif label_path3.exists():

        label_path = label_path3

    elif label_path4.exists():

        label_path = label_path4

    else:

        label_path = None

    # --------------------------------------------------------
    # 类别数
    # --------------------------------------------------------

    num_cls = (
        len(classes)
        if classes
        else 80
    )

    color_map = build_color_map(
        num_cls,
        min_s=0.70,
        max_s=1.00,
        min_v=0.90,
        max_v=1.00,
    )

    # --------------------------------------------------------
    # 绘制
    # --------------------------------------------------------

    out_img, cnt = draw_labels_on_image(
        img_path,
        label_path,
        classes=classes,
        color_map=color_map,
        label_to_id=label_to_id,
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    if out_path:

        save_image(
            out_img,
            out_path,
        )

    # --------------------------------------------------------
    # 显示
    # --------------------------------------------------------

    if show:

        display_img = out_img.copy()

        h, w = display_img.shape[:2]

        # 最大显示尺寸
        max_w = 1920
        max_h = 1080

        scale = min(
            max_w / w,
            max_h / h,
            1.0,
        )

        if scale < 1.0:

            display_img = cv2.resize(
                display_img,
                (
                    int(w * scale),
                    int(h * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(
            "draw_img",
            display_img,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(
        f"[DONE] processed image "
        f"with {cnt} objects"
    )


# ============================================================
# 打印颜色表
# ============================================================

def print_color_map(
    color_map,
    classes=None,
):
    """
    打印类别颜色。
    """

    print()
    print("=" * 70)
    print("Color Map")
    print("=" * 70)

    for class_id, color in color_map.items():

        if (
            classes is not None
            and 0 <= class_id < len(classes)
        ):
            name = classes[class_id]
        else:
            name = str(class_id)

        print(
            f"{class_id:3d} | "
            f"{name:<20s} | "
            f"BGR={color}"
        )

    print("=" * 70)
    print()


# ============================================================
# 参数
# ============================================================

def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Draw YOLO / LabelMe labels "
            "on images"
        )
    )

    parser.add_argument(
        "--images",
        default=r"E:\test_dino\images",
        help=(
            "images folder "
            "or single image path"
        ),
    )

    parser.add_argument(
        "--labels",
        default=r"E:\test_dino\jsons",
        help=(
            "labels folder "
            "(matching image basenames) "
            "or label file path"
        ),
    )

    parser.add_argument(
        "--out",
        default=r"E:\test_dino\outputs",
        help=(
            "output folder "
            "or single output image path"
        ),
    )

    parser.add_argument(
        "--classes",
        default=r"D:\DataSets\MyRoad\classes.txt",
        help=(
            "optional classes.txt "
            "(one class per line)"
        ),
    )

    parser.add_argument(
        "--save_undetected",
        action="store_true",
        help=(
            "also save images "
            "without labels"
        ),
    )

    parser.add_argument(
        "--ext",
        nargs="*",
        default=[
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
        ],
        help=(
            "image extensions"
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help=(
            "show image window "
            "(single image mode only)"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help=(
            "number of threads "
            "(default: 32)"
        ),
    )

    parser.add_argument(
        "--print_colors",
        action="store_true",
        help=(
            "print generated color map"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    images = Path(
        args.images
    )

    labels = Path(
        args.labels
    )

    out = Path(
        args.out
    )

    # ========================================================
    # 单张图片
    # ========================================================

    if images.is_file():

        process_single_image(
            images,
            labels,
            out,
            classes_path=args.classes,
            show=args.show,
        )

    # ========================================================
    # 文件夹
    # ========================================================

    else:

        classes = (
            load_classes(args.classes)
            if args.classes
            else None
        )

        if args.print_colors:

            num_classes = (
                len(classes)
                if classes
                else 80
            )

            color_map = build_color_map(
                num_classes,
                min_s=0.70,
                max_s=1.00,
                min_v=0.90,
                max_v=1.00,
            )

            print_color_map(
                color_map,
                classes,
            )

        process_folder(
            images,
            labels,
            out,
            classes_path=args.classes,
            save_undetected=args.save_undetected,
            ext_list=[
                e.lower()
                for e in args.ext
            ],
            num_workers=args.workers,
        )
