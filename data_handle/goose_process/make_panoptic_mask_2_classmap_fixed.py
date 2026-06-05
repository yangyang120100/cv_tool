#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全景分割掩码和json标签生成
GOOSE / 自定义数据集 -> COCO Panoptic / OneFormer 可训练数据（类别 id 直接沿用 goose_classmap.json）

输入：
  --image_dir        原图目录
  --semseg_dir       语义分割 mask 目录（单通道类别索引图）
  --inst_dir         实例分割 mask 目录（单文件多值图 / RGB ID 图 / 多个二值图）
  --out_dir          输出目录
  --split_file       可选：当前 split 的图像列表（每行一个 basename 或文件名）
  --categories_json  可选：类别配置文件，支持 raw_id、name、isthing、supercategory

输出：
  out_dir/
    panoptic_<split>.json
    panoptic_masks/
      xxx.png
    vis/
      xxx_vis.png
    debug_report.json

说明：
  1) 生成的 panoptic mask PNG 使用 RGB 编码，等价于 panopticapi 的 id2rgb 逻辑。
  2) category_id 默认重映射为连续正整数，从 1 开始，避免 0 冲突。
  3) thing 类的 segment_id = category_id * label_divisor + instance_index
     stuff 类的 segment_id = category_id * label_divisor + 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# -----------------------------
# 基础工具
# -----------------------------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_image_np(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path))


def is_image_file(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def rgb_from_id_map(id_map: np.ndarray) -> np.ndarray:
    """
    将整数 id map 编码为 RGB 图。
    与 panopticapi id2rgb 兼容：
      R = id % 256
      G = (id // 256) % 256
      B = (id // 65536) % 256
    """
    id_map = id_map.astype(np.int64, copy=False)
    rgb = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (id_map & 255).astype(np.uint8)
    rgb[..., 1] = ((id_map >> 8) & 255).astype(np.uint8)
    rgb[..., 2] = ((id_map >> 16) & 255).astype(np.uint8)
    return rgb


def id_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    RGB -> 整数 id。
    """
    rgb = rgb.astype(np.int64, copy=False)
    return rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)


def save_panoptic_png(id_map: np.ndarray, save_path: str | Path) -> None:
    rgb = rgb_from_id_map(id_map)
    Image.fromarray(rgb).save(save_path, format="PNG")


def compute_bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(np.uint8).sum())


def read_semantic_mask(path: str | Path) -> np.ndarray:
    """
    语义 mask：默认单通道类别索引图。
    如果是 RGB/彩色图，优先取单通道，否则报错。
    """
    arr = load_image_np(path)
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3:
        # 如果用户误存成 RGB，但本质是灰度图，PIL 常常会读成 (H,W,3)
        # 这里先取一致的灰度转换，避免静默出错。
        return np.array(Image.fromarray(arr).convert("L"), dtype=np.int32)
    raise ValueError(f"Unsupported semantic mask shape: {arr.shape} at {path}")


def read_instance_mask_as_id_map(path: str | Path) -> np.ndarray:
    """
    实例 mask 支持：
      1) 单通道多值图：每个像素值是 instance id
      2) RGB 编码 ID 图：按 RGB->id 解码
      3) 二值图：由调用方当作单个实例处理
    """
    arr = load_image_np(path)

    if arr.ndim == 2:
        return arr.astype(np.int64)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return id_from_rgb(arr).astype(np.int64)

    raise ValueError(f"Unsupported instance mask shape: {arr.shape} at {path}")


def read_binary_instance_mask(path: str | Path) -> np.ndarray:
    arr = load_image_np(path)
    if arr.ndim == 2:
        return arr != 0
    if arr.ndim == 3:
        # 彩色二值图也按非零视为前景
        return np.any(arr != 0, axis=2)
    raise ValueError(f"Unsupported binary instance mask shape: {arr.shape} at {path}")


def find_image_path(image_dir: str | Path, stem: str) -> Optional[Path]:
    image_dir = Path(image_dir)
    candidates = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            candidates.append(p)
    if candidates:
        return candidates[0]

    # 兜底：目录里找同 stem
    for p in image_dir.iterdir():
        if p.is_file() and p.stem == stem and is_image_file(p.name):
            return p
    return None


# -----------------------------
# 类别配置
# -----------------------------

@dataclass
class CategorySpec:
    raw_id: int
    category_id: int
    name: str
    isthing: Optional[int] = None
    supercategory: str = "none"


def load_category_specs(categories_json: Optional[str]) -> Dict[int, CategorySpec]:
    """
    支持两种格式：
      1) {"categories":[{...}, {...}]}
      2) [{...}, {...}]

    推荐 goose_classmap.json 显式提供：
      - raw_id: 语义 mask 里的原始类别值
      - id:     输出到 panoptic JSON / OneFormer 的 category_id

    如果缺少 id，则默认使用 raw_id，避免对类别重新编码。
    """
    if not categories_json:
        return {}

    with open(categories_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "categories" in data:
        items = data["categories"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("categories_json 格式不对，应该是 list 或 {'categories': [...]}")

    specs: Dict[int, CategorySpec] = {}
    seen_cat_ids: Dict[int, int] = {}
    for item in items:
        raw_id = item.get("raw_id", item.get("semantic_id", item.get("id", None)))
        if raw_id is None:
            raise ValueError(f"类别项缺少 raw_id/id: {item}")
        raw_id = int(raw_id)

        category_id = int(item.get("id", raw_id))

        if category_id in seen_cat_ids and seen_cat_ids[category_id] != raw_id:
            raise ValueError(
                f"发现多个 raw_id 映射到同一个 category_id={category_id}: "
                f"{seen_cat_ids[category_id]} vs {raw_id}"
            )
        seen_cat_ids[category_id] = raw_id

        specs[raw_id] = CategorySpec(
            raw_id=raw_id,
            category_id=category_id,
            name=str(item.get("name", f"class_{raw_id}")),
            isthing=(None if item.get("isthing", None) is None else int(item["isthing"])),
            supercategory=str(item.get("supercategory", "none")),
        )
    return specs


# -----------------------------
# 实例读取
# -----------------------------

def collect_instances_for_image(
    image_stem: str,
    inst_dir: str | Path,
    sem_mask: np.ndarray,
) -> List[dict]:
    """
    返回：
      [
        {
          "mask": bool ndarray,
          "source_instance_id": int,
          "raw_category_id": int,
        },
        ...
      ]

    自动支持两类实例标注：
      A) <stem>.png / .tif / ... 单文件 id map
      B) 多个二值 mask 文件，文件名中包含 stem
    """
    inst_dir = Path(inst_dir)
    instances: List[dict] = []

    # A. 先尝试找一个与 stem 同名的单文件实例图
    for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"):
        p = inst_dir / f"{image_stem}{ext}"
        if p.exists():
            id_map = read_instance_mask_as_id_map(p)
            unique_ids = [int(x) for x in np.unique(id_map) if int(x) != 0]
            for inst_id in unique_ids:
                m = id_map == inst_id
                if m.any():
                    raw_cat = majority_vote_category(sem_mask, m)
                    instances.append(
                        {
                            "mask": m,
                            "source_instance_id": int(inst_id),
                            "raw_category_id": int(raw_cat),
                        }
                    )
            return instances

    # B. 多文件二值 mask
    #    规则：文件名包含 image_stem，且不是语义 mask
    all_files = sorted([p for p in inst_dir.iterdir() if p.is_file() and is_image_file(p.name)])
    for p in all_files:
        if image_stem not in p.stem:
            continue

        # 尽量排除明显不是实例文件的命名
        low = p.stem.lower()
        if any(x in low for x in ("sem", "semantic", "label", "labels")):
            continue

        m = read_binary_instance_mask(p)
        if not m.any():
            continue

        raw_cat = majority_vote_category(sem_mask, m)
        # 尝试从文件名里解析实例序号，否则顺序编号
        inst_id = parse_int_from_string(p.stem)
        if inst_id is None:
            inst_id = len(instances) + 1

        instances.append(
            {
                "mask": m,
                "source_instance_id": int(inst_id),
                "raw_category_id": int(raw_cat),
            }
        )

    return instances


def parse_int_from_string(s: str) -> Optional[int]:
    m = re.findall(r"\d+", s)
    if not m:
        return None
    return int(m[-1])


def majority_vote_category(sem_mask: np.ndarray, inst_mask: np.ndarray, ignore_labels=(255,)) -> int:
    vals = sem_mask[inst_mask]
    vals = vals[np.isin(vals, list(ignore_labels), invert=True)]
    if vals.size == 0:
        return -1
    unique, counts = np.unique(vals.astype(np.int64), return_counts=True)
    return int(unique[np.argmax(counts)])


# -----------------------------
# 主合并逻辑
# -----------------------------

def build_raw_to_cat_id(
    all_raw_ids: List[int],
    category_specs: Dict[int, CategorySpec],
    exclude_raw_ids: Tuple[int, ...] = (255,),
    zero_is_background: bool = True,
) -> Tuple[Dict[int, int], List[dict]]:
    """
    直接使用 goose_classmap.json 中定义的 category_id，不做重新编码。

    返回：
      raw_to_cat: 语义 mask 原始值 -> panoptic category_id
      categories: 供 panoptic json 使用的 categories 列表

    这可以保证：
      1) 语义 mask 的类别值和 panoptic 的 category_id 一一对应
      2) OneFormer 训练时不会因为“连续重编码”导致类别错位
    """
    raw_ids = []
    for rid in sorted(set(int(x) for x in all_raw_ids)):
        if rid in exclude_raw_ids:
            continue
        if zero_is_background and rid == 0:
            continue
        raw_ids.append(rid)

    raw_to_cat: Dict[int, int] = {}
    categories: List[dict] = []

    for raw_id in raw_ids:
        spec = category_specs.get(raw_id)
        if spec is not None:
            cat_id = int(spec.category_id)
            name = spec.name
            supercategory = spec.supercategory
            isthing = 0 if spec.isthing is None else int(spec.isthing)
        else:
            # 若 classmap 没覆盖到某个 raw_id，保留原值，避免再次重编码。
            cat_id = int(raw_id)
            name = f"class_{raw_id}"
            supercategory = "none"
            isthing = 0
            warnings.warn(
                f"raw_id={raw_id} 未在 goose_classmap.json 中找到，"
                f"将使用原始 id 作为 category_id。请检查类别映射是否完整。"
            )

        raw_to_cat[raw_id] = cat_id
        categories.append(
            {
                "id": cat_id,
                "name": name,
                "supercategory": supercategory,
                "isthing": isthing,
                "raw_id": raw_id,
            }
        )

    categories = sorted(categories, key=lambda x: x["id"])
    return raw_to_cat, categories


def build_panoptic_for_image(
    sem_mask: np.ndarray,
    instances: List[dict],
    raw_to_cat_id: Dict[int, int],
    category_specs: Dict[int, CategorySpec],
    label_divisor: int = 1000,
    min_instance_area: int = 0,
    keep_semantic_only: bool = True,
    ignore_label: int = 255,
) -> Tuple[np.ndarray, List[dict], Dict[int, int]]:
    """
    返回：
      pan_id_map: int32
      segments_info: list[dict]
      raw_id_isthing_inferred: {raw_raw_id: 0/1}
    """
    h, w = sem_mask.shape
    pan_id_map = np.zeros((h, w), dtype=np.int32)
    occupied = np.zeros((h, w), dtype=bool)

    segments_info: List[dict] = []
    raw_id_isthing_inferred: Dict[int, int] = {}

    # 先收集 instance 里出现过的 raw 类别，作为 thing 的强证据
    raw_instance_count: Dict[int, int] = {}
    for inst in instances:
        raw_cat = int(inst["raw_category_id"])
        if raw_cat == -1 or raw_cat == ignore_label:
            continue
        raw_instance_count[raw_cat] = raw_instance_count.get(raw_cat, 0) + 1

    # 处理 instance，优先级高于 semantic
    # 为了避免相互覆盖，按面积从大到小处理更稳定
    instances_sorted = sorted(instances, key=lambda x: int(x["mask"].sum()), reverse=True)

    next_inst_idx_per_cat: Dict[int, int] = {}

    for inst in instances_sorted:
        inst_mask = inst["mask"].astype(bool)
        inst_mask = inst_mask & (~occupied)
        area = mask_area(inst_mask)
        if area < min_instance_area:
            continue

        raw_cat = int(inst["raw_category_id"])
        if raw_cat < 0 or raw_cat == ignore_label:
            continue
        if raw_cat not in raw_to_cat_id:
            # 语义里没这个类，直接跳过
            continue

        cat_id = int(raw_to_cat_id[raw_cat])
        next_inst_idx_per_cat.setdefault(cat_id, 1)
        inst_idx = int(next_inst_idx_per_cat[cat_id])
        next_inst_idx_per_cat[cat_id] += 1

        segment_id = int(cat_id * label_divisor + inst_idx)

        pan_id_map[inst_mask] = segment_id
        occupied |= inst_mask

        bbox = compute_bbox_from_mask(inst_mask)

        segments_info.append(
            {
                "id": segment_id,
                "category_id": cat_id,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "instance_id": inst_idx,
                "raw_category_id": raw_cat,
            }
        )
        raw_id_isthing_inferred[raw_cat] = 1

    # 处理剩余 semantic 区域
    for raw_cat in sorted(np.unique(sem_mask).astype(int).tolist()):
        if raw_cat == ignore_label:
            continue
        if raw_cat == 255:
            # 默认把 0 当背景/void；如果你的数据里 0 是有效 stuff，需要在这里改逻辑
            continue
        if raw_cat not in raw_to_cat_id:
            continue

        remain = (sem_mask == raw_cat) & (~occupied)
        area = mask_area(remain)
        if area == 0:
            continue

        cat_id = int(raw_to_cat_id[raw_cat])

        # 如果该类有 instance 证据，说明是 thing；
        # 否则按 stuff 处理。
        isthing = 1 if raw_cat in raw_instance_count else 0
        raw_id_isthing_inferred[raw_cat] = isthing

        if isthing == 1:
            if not keep_semantic_only:
                continue
            inst_idx = 0
            segment_id = int(cat_id * label_divisor + inst_idx)
            iscrowd = 1
        else:
            inst_idx = 0
            segment_id = int(cat_id * label_divisor + inst_idx)
            iscrowd = 0

        pan_id_map[remain] = segment_id
        occupied |= remain

        bbox = compute_bbox_from_mask(remain)
        segments_info.append(
            {
                "id": segment_id,
                "category_id": cat_id,
                "area": area,
                "bbox": bbox,
                "iscrowd": iscrowd,
                "instance_id": inst_idx,
                "raw_category_id": raw_cat,
            }
        )

    return pan_id_map, segments_info, raw_id_isthing_inferred


# -----------------------------
# 校验与可视化
# -----------------------------

def validate_panoptic_roundtrip(
    pan_id_map: np.ndarray,
    segments_info: List[dict],
    save_path: str | Path,
) -> Dict[str, int]:
    """
    1) 保存的 PNG 再读回来，检查 id 是否一致
    2) 检查 segments_info 中每个 id 在图中都存在
    """
    save_panoptic_png(pan_id_map, save_path)
    decoded = id_from_rgb(load_image_np(save_path)).astype(np.int32)

    same = int(np.array_equal(decoded, pan_id_map))
    seg_ids_in_img = set(int(x) for x in np.unique(decoded).tolist())
    seg_ids_in_json = set(int(s["id"]) for s in segments_info)

    missing_in_img = len(seg_ids_in_json - seg_ids_in_img)
    extra_in_img = len(seg_ids_in_img - seg_ids_in_json - {0})

    return {
        "roundtrip_equal": same,
        "missing_segment_ids_in_img": missing_in_img,
        "extra_ids_not_in_json": extra_in_img,
    }


def random_color_for_id(x: int) -> Tuple[int, int, int]:
    """
    生成稳定伪随机颜色，用于可视化。
    """
    x = int(x)
    r = (37 * x + 17) % 255
    g = (29 * x + 71) % 255
    b = (53 * x + 101) % 255
    return int(r), int(g), int(b)


def save_visualization(
    image_path: str | Path,
    pan_id_map: np.ndarray,
    segments_info: List[dict],
    out_path: str | Path,
) -> None:
    """
    保存一个简单的可视化：原图 + panoptic 伪彩色。
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    vis = img_np.copy()

    # 给每个 segment 上色
    id_to_color = {0: (0, 0, 0)}
    for seg in segments_info:
        sid = int(seg["id"])
        id_to_color[sid] = random_color_for_id(sid)

    colored = np.zeros_like(vis)
    for sid in np.unique(pan_id_map):
        sid = int(sid)
        color = id_to_color.get(sid, random_color_for_id(sid))
        colored[pan_id_map == sid] = color

    # 简单 alpha blend
    alpha = 0.45
    blended = (vis * (1 - alpha) + colored * alpha).astype(np.uint8)

    canvas = Image.new("RGB", (img.width * 2, img.height))
    canvas.paste(Image.fromarray(vis), (0, 0))
    canvas.paste(Image.fromarray(blended), (img.width, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "image", fill=(255, 255, 255))
    draw.text((img.width + 10, 10), "panoptic overlay", fill=(255, 255, 255))
    canvas.save(out_path)


# -----------------------------
# 数据集生成
# -----------------------------

def list_images_from_split(
    image_dir: str | Path,
    split_file: Optional[str],
) -> List[Path]:
    image_dir = Path(image_dir)

    if split_file:
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        paths = []
        for line in lines:
            stem = Path(line).stem
            p = find_image_path(image_dir, stem)
            if p is None:
                raise FileNotFoundError(f"split_file 中的图像找不到对应原图: {line}")
            paths.append(p)
        return paths

    return sorted([p for p in image_dir.iterdir() if p.is_file() and is_image_file(p.name)])


def generate_panoptic_dataset(
    image_dir: str,
    semseg_dir: str,
    inst_dir: str,
    out_dir: str,
    split_name: str = "train",
    split_file: Optional[str] = None,
    categories_json: Optional[str] = None,
    label_divisor: int = 1000,
    min_instance_area: int = 0,
    keep_semantic_only: bool = True,
    ignore_label: int = 255,
    zero_is_background: bool = True,
    vis_n: int = 5,
) -> None:
    ensure_dir(out_dir)
    pan_root = Path(out_dir) / "panoptic_masks"
    vis_root = Path(out_dir) / "vis"
    ensure_dir(pan_root)
    ensure_dir(vis_root)

    category_specs = load_category_specs(categories_json)

    image_paths = list_images_from_split(image_dir, split_file)
    if not image_paths:
        raise RuntimeError("没有找到任何图像。请检查 image_dir 或 split_file。")

    # 先扫描所有 semantic raw ids，构建 raw->category_id 映射。
    # 这里不做连续重编码，而是直接读取 goose_classmap.json 中定义的 id。
    all_raw_ids = set()
    for img_path in tqdm(image_paths, desc="Scanning semantic ids"):
        sem_path = Path(semseg_dir) / f"{img_path.stem}.png"
        if not sem_path.exists():
            # 允许不同扩展名
            cand = None
            for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"):
                p = Path(semseg_dir) / f"{img_path.stem}{ext}"
                if p.exists():
                    cand = p
                    break
            if cand is None:
                raise FileNotFoundError(f"找不到语义 mask: {img_path.stem}")
            sem_path = cand

        sem = read_semantic_mask(sem_path)
        vals = np.unique(sem).astype(int).tolist()
        all_raw_ids.update(vals)

    raw_to_cat_id, categories = build_raw_to_cat_id(
        sorted(all_raw_ids),
        category_specs=category_specs,
        exclude_raw_ids=(ignore_label,),
        zero_is_background=zero_is_background,
    )

    images = []
    annotations = []
    debug_report = {
        "num_images": 0,
        "num_annotations": 0,
        "num_segments": 0,
        "roundtrip_errors": [],
        "missing_semantic_masks": [],
        "missing_instance_masks": [],
    }

    # 统计 thing/stuff
    raw_id_isthing_final: Dict[int, int] = {}
    image_id = 1

    for idx, img_path in enumerate(tqdm(image_paths, desc=f"Building {split_name}")):
        stem = img_path.stem

        sem_path = None
        for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"):
            p = Path(semseg_dir) / f"{stem}{ext}"
            if p.exists():
                sem_path = p
                break
        if sem_path is None:
            debug_report["missing_semantic_masks"].append(stem)
            continue

        sem = read_semantic_mask(sem_path)

        insts = collect_instances_for_image(stem, inst_dir, sem)

        pan_id_map, segments_info, inferred_isthing = build_panoptic_for_image(
            sem_mask=sem,
            instances=insts,
            raw_to_cat_id=raw_to_cat_id,
            category_specs=category_specs,
            label_divisor=label_divisor,
            min_instance_area=min_instance_area,
            keep_semantic_only=keep_semantic_only,
            ignore_label=ignore_label,
        )

        for k, v in inferred_isthing.items():
            # 只要有实例证据，就把它标为 thing
            if k not in raw_id_isthing_final:
                raw_id_isthing_final[k] = int(v)
            else:
                raw_id_isthing_final[k] = max(raw_id_isthing_final[k], int(v))

        pan_file = f"{stem}.png"
        pan_path = pan_root / pan_file

        val = validate_panoptic_roundtrip(pan_id_map, segments_info, pan_path)
        if not val["roundtrip_equal"]:
            debug_report["roundtrip_errors"].append(
                {
                    "stem": stem,
                    "roundtrip_equal": val["roundtrip_equal"],
                    "missing_segment_ids_in_img": val["missing_segment_ids_in_img"],
                    "extra_ids_not_in_json": val["extra_ids_not_in_json"],
                }
            )

        if idx < vis_n:
            vis_path = vis_root / f"{stem}_vis.png"
            save_visualization(img_path, pan_id_map, segments_info, vis_path)

        width, height = Image.open(img_path).size
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "height": height,
                "width": width,
            }
        )
        annotations.append(
            {
                "image_id": image_id,
                "file_name": pan_file,
                "segments_info": [
                    {
                        "id": int(s["id"]),
                        "category_id": int(s["category_id"]),
                        "area": int(s["area"]),
                        "bbox": [int(x) for x in s["bbox"]],
                        "iscrowd": int(s["iscrowd"]),
                    }
                    for s in segments_info
                ],
            }
        )

        debug_report["num_images"] += 1
        debug_report["num_annotations"] += 1
        debug_report["num_segments"] += len(segments_info)

        image_id += 1

    # 更新 categories 的 isthing
    for cat in categories:
        raw_id = int(cat["raw_id"])
        if raw_id in category_specs and category_specs[raw_id].isthing is not None:
            cat["isthing"] = int(category_specs[raw_id].isthing)
        else:
            cat["isthing"] = int(raw_id_isthing_final.get(raw_id, 0))

        # raw_id 不是 COCO panoptic 必需字段，但保留在输出 json 里很有用
        # 需要严格 COCO 兼容时可以删掉它
        cat["id"] = int(cat["id"])
        cat["raw_id"] = int(cat["raw_id"])

    # 按 id 排序
    categories = sorted(categories, key=lambda x: x["id"])

    panoptic_json = {
        "info": {
            "description": f"{split_name} panoptic annotations converted for OneFormer",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": int(c["id"]),
                "name": c["name"],
                "supercategory": c["supercategory"],
                "isthing": int(c["isthing"]),
            }
            for c in categories
        ],
    }

    json_path = Path(out_dir) / f"panoptic_{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(panoptic_json, f, ensure_ascii=False, indent=2)

    report_path = Path(out_dir) / "debug_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(debug_report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Panoptic JSON saved to: {json_path}")
    print(f"[OK] Debug report saved to: {report_path}")
    print(f"[OK] Panoptic masks saved under: {pan_root}")
    print(f"[OK] Visualizations saved under: {vis_root}")


# -----------------------------
# Detectron2 / OneFormer 注册示例
# -----------------------------

def print_registration_example(
    dataset_name: str,
    image_dir: str,
    panoptic_json: str,
    panoptic_root: str,
) -> None:
    print(
        f"""
# Detectron2 / OneFormer 注册示例

from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data import MetadataCatalog

register_coco_panoptic_separated(
    "{dataset_name}",
    {{
        "image_root": r"{image_dir}",
        "panoptic_root": r"{panoptic_root}",
        "panoptic_json": r"{panoptic_json}",
    }},
    {{
        "image_root": r"{image_dir}",
        "panoptic_root": r"{panoptic_root}",
        "panoptic_json": r"{panoptic_json}",
    }},
    metadata={{}},
)

meta = MetadataCatalog.get("{dataset_name}")
meta.evaluator_type = "coco_panoptic_seg"
"""
    )


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser("Generate COCO panoptic labels for OneFormer")

    p.add_argument("--image_dir", type=str, default=r"D:\DataBase\road\goose\train\images")
    p.add_argument("--semseg_dir", type=str, default=r"D:\DataBase\road\goose\train\labels\semantic")
    p.add_argument("--inst_dir", type=str, default=r"D:\DataBase\road\goose\train\labels\instanceids",help="实例分割图像路径")
    p.add_argument("--out_dir", type=str, default=r"D:\DataBase\road\goose\train\labels")

    p.add_argument("--split_name", type=str, default="val")
    p.add_argument("--split_file", type=str, default=None)
    p.add_argument("--categories_json", type=str, default=r"D:\Projects\Scripting_tool\data_handle\goose_process\goose_classmap.json")

    p.add_argument("--label_divisor", type=int, default=1000)
    p.add_argument("--min_instance_area", type=int, default=0)
    p.add_argument("--keep_semantic_only", action="store_true", default=True)
    p.add_argument("--drop_semantic_only", action="store_true", default=False)

    p.add_argument("--ignore_label", type=int, default=255)
    p.add_argument("--zero_is_background", action="store_true", default=False)
    p.add_argument("--zero_is_valid_class", action="store_true", default=True)

    p.add_argument("--vis_n", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    keep_semantic_only = args.keep_semantic_only and (not args.drop_semantic_only)
    zero_is_background = (not args.zero_is_valid_class)

    generate_panoptic_dataset(
        image_dir=args.image_dir,
        semseg_dir=args.semseg_dir,
        inst_dir=args.inst_dir,
        out_dir=args.out_dir,
        split_name=args.split_name,
        split_file=args.split_file,
        categories_json=args.categories_json,
        label_divisor=args.label_divisor,
        min_instance_area=args.min_instance_area,
        keep_semantic_only=keep_semantic_only,
        ignore_label=args.ignore_label,
        zero_is_background=zero_is_background,
        vis_n=args.vis_n,
    )

    print_registration_example(
        dataset_name=f"goose_panoptic_{args.split_name}",
        image_dir=args.image_dir,
        panoptic_json=str(Path(args.out_dir) / f"panoptic_{args.split_name}.json"),
        panoptic_root=str(Path(args.out_dir) / "panoptic_masks"),
    )


if __name__ == "__main__":
    main()