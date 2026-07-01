import cv2
import json
import numpy as np
from pathlib import Path
import os


ALL_CLASSES = ['background', 'asphalt-road', 'concrete-pavement', 'brick-road', 'dirt', 'gravel', 'inland-water', 'standing-water', 'woods', 'tall-vegetation', 'low-vegetation', 'roof', 'wall', 'fence', 'fence-post', 'snow', 'concrete-pole', 'angle-steel-tower', 'steel-tube-tower', 'conductor', 'sky', 'obstacle', 'car', 'bus', 'work-vehicle', 'large-vehicle', 'bicycle', 'person', 'manhole-cover', 'distant-building', 'traffic-light', 'house', 'tricycle', 'distribution box']

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

    image_rel_path = os.path.relpath(
        image_path,
        start=output_json.parent
    ).replace("\\", "/")

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


def batch_convert(
        mask_dir,
        image_dir,
        json_dir,
        min_area=50,
        epsilon_ratio=0.003
):
    mask_dir = Path(mask_dir)
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)

    json_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_map = {}

    for img_path in image_dir.iterdir():

        if img_path.suffix.lower() in IMAGE_SUFFIX:
            image_map[
                img_path.stem
            ] = img_path

    success = 0
    failed = 0
    skipped = 0

    mask_files = sorted(
        mask_dir.glob("*.png")
    )

    print(
        f"find mask: {len(mask_files)}"
    )

    for mask_path in mask_files:

        stem = mask_path.stem

        image_path = image_map.get(stem)

        if image_path is None:

            print(
                f"[skip] image not found: {stem}"
            )

            skipped += 1
            continue

        output_json = (
            json_dir /
            f"{stem}.json"
        )

        try:

            mask_to_labelme(
                mask_path,
                image_path,
                output_json,
                min_area=min_area,
                epsilon_ratio=epsilon_ratio
            )

            success += 1

            print(
                f"[ok] {stem}"
            )

        except Exception as e:

            failed += 1

            print(
                f"[fail] {stem}: {e}"
            )

    print("\n==========")
    print(
        f"success : {success}"
    )
    print(
        f"failed  : {failed}"
    )
    print(
        f"skipped : {skipped}"
    )
    print("==========")


if __name__ == "__main__":

    batch_convert(
        mask_dir=r"E:\cable_data\masks",
        image_dir=r"E:\cable_data\images",
        json_dir=r"E:\cable_data\mask_jsons",
        min_area=4000,
        epsilon_ratio=0.001
    )