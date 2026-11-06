#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Panoptic Checker
- Faster RGB->ID conversion
- Faster bbox computation
- More detailed validation
- Dataset summary report
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm


def rgb2id_bgr(img):
    return (
        img[:, :, 2].astype(np.int64)
        + 256 * img[:, :, 1].astype(np.int64)
        + 256 * 256 * img[:, :, 0].astype(np.int64)
    )


def compute_bbox_fast(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x, y, w, h = cv2.boundingRect(
        np.column_stack((xs, ys)).astype(np.int32)
    )
    return [int(x), int(y), int(w), int(h)]


def check_single_image(png_path, ann, category_dict):
    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    panoptic = rgb2id_bgr(img)

    h, w = panoptic.shape

    errors = []
    warnings = []

    png_ids = set(np.unique(panoptic))
    png_ids.discard(0)

    json_ids = set()
    seen_ids = set()

    for seg in ann["segments_info"]:
        sid = int(seg["id"])

        if sid in seen_ids:
            errors.append(f"duplicate segment id in json: {sid}")
        seen_ids.add(sid)

        json_ids.add(sid)

        cid = int(seg["category_id"])

        if cid not in category_dict:
            errors.append(f"id={sid} unknown category_id={cid}")
            continue

        mask = (panoptic == sid)

        if not mask.any():
            errors.append(f"id={sid} exists in json but not png")
            continue

        area = int(mask.sum())
        if area != seg["area"]:
            errors.append(
                f"id={sid} area mismatch json={seg['area']} png={area}"
            )

        bbox = compute_bbox_fast(mask)

        if bbox != seg["bbox"]:
            warnings.append(
                f"id={sid} bbox mismatch json={seg['bbox']} png={bbox}"
            )

        x, y, bw, bh = bbox
        if x < 0 or y < 0 or x + bw > w or y + bh > h:
            errors.append(f"id={sid} bbox out of image")

        if category_dict[cid]["isthing"]:
            cc_num = cv2.connectedComponents(
                mask.astype(np.uint8), connectivity=8
            )[0] - 1

            if cc_num > 1:
                warnings.append(
                    f"id={sid} thing split into {cc_num} components"
                )

    extra_ids = png_ids - json_ids
    missing_ids = json_ids - png_ids

    if extra_ids:
        errors.append(
            f"png contains {len(extra_ids)} extra ids: "
            f"{sorted(list(extra_ids))[:10]}"
        )

    if missing_ids:
        errors.append(
            f"json contains {len(missing_ids)} missing ids: "
            f"{sorted(list(missing_ids))[:10]}"
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "num_png_segments": len(png_ids),
        "num_json_segments": len(json_ids),
    }


def dataset_statistics(data):
    category_count = Counter()
    thing_count = Counter()

    cat_info = {c["id"]: c for c in data["categories"]}

    for ann in data["annotations"]:
        for seg in ann["segments_info"]:
            cid = seg["category_id"]
            category_count[cid] += 1

            if cat_info[cid]["isthing"]:
                thing_count[cid] += 1

    print("\\n===== CATEGORY STATISTICS =====")

    for cid in sorted(category_count):
        print(
            f"{cid:4d} "
            f"{cat_info[cid]['name'][:30]:30s} "
            f"segments={category_count[cid]} "
            f"thing={thing_count[cid]}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--panoptic_json", default=r"D:\DataBase\road\goose\train\labels\panoptic_val.json")
    parser.add_argument("--panoptic_root", default=r"D:\DataBase\road\goose\train\labels\panoptic_masks")
    parser.add_argument("--save_report", default=None)

    args = parser.parse_args()

    with open(args.panoptic_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_dict = {
        c["id"]: c for c in data["categories"]
    }

    dataset_statistics(data)

    total_errors = 0
    total_warnings = 0

    image_reports = []

    print("\\n===== CHECKING IMAGES =====")

    for ann in tqdm(data["annotations"]):

        png_path = Path(args.panoptic_root) / ann["file_name"]

        if not png_path.exists():
            total_errors += 1
            image_reports.append({
                "file": ann["file_name"],
                "errors": ["missing png"]
            })
            continue

        result = check_single_image(
            png_path,
            ann,
            category_dict
        )

        total_errors += len(result["errors"])
        total_warnings += len(result["warnings"])

        if result["errors"] or result["warnings"]:
            image_reports.append({
                "file": ann["file_name"],
                "errors": result["errors"],
                "warnings": result["warnings"]
            })

    print("\\n===== SUMMARY =====")
    print("images:", len(data["annotations"]))
    print("errors:", total_errors)
    print("warnings:", total_warnings)

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(
                image_reports,
                f,
                ensure_ascii=False,
                indent=2
            )
        print("report saved:", args.save_report)

    print("\\nPASS ✓" if total_errors == 0 else "\\nFAIL ✗")


if __name__ == "__main__":
    main()
