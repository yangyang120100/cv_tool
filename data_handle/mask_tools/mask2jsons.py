import cv2
import json
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

ALL_CLASSES =['background', 'asphalt-road', 'concrete-pavement', 'brick-road', 'dirt', 'gravel', 'inland-water', 'standing-water', 'woods', 'tall-vegetation', 'low-vegetation', 'roof', 'wall', 'fence', 'fence-post', 'snow', 'concrete-pole', 'angle-steel-tower', 'steel-tube-tower', 'conductor', 'sky', 'obstacle', 'car', 'bus', 'work-vehicle', 'large-vehicle', 'bicycle', 'person', 'manhole-cover', 'distant-building', 'traffic-light', 'house', 'tricycle','lead_wire']

CLASS_MAP = {
    i: j for i, j in enumerate(ALL_CLASSES)
}

IMAGE_SUFFIX = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

BRIDGE_WIDTH = 2

BRIDGE_EXTENSION = 4

BRIDGED_MAX_EPSILON_PX = 0.75

MAX_BRIDGE_SAMPLE_POINTS = 2000

MIN_HOLE_AREA = 1.0


# contour 采样
def sample_contour_points(
        contour,
        max_points=2000
):
    """
    对 contour 点进行均匀采样。
    用于寻找 hole 与 outer contour 的最近点。
    """

    points = (
        contour
        .reshape(-1, 2)
        .astype(np.float32)
    )

    n = len(points)

    if n <= max_points:
        return points

    indices = np.linspace(
        0,
        n - 1,
        max_points,
        dtype=np.int32
    )

    return points[indices]


# 寻找两个 contour 的最近点

def find_closest_points(
        contour_a,
        contour_b,
        max_sample_points=2000,
        chunk_size=256
):
    """
    在两个 contour 之间寻找近似欧氏距离最近的点。
    返回：

        point_a
        point_b
        distance
    """

    points_a = sample_contour_points(
        contour_a,
        max_sample_points
    )

    points_b = sample_contour_points(
        contour_b,
        max_sample_points
    )

    best_distance_sq = float("inf")
    best_a = None
    best_b = None

    for start in range(
        0,
        len(points_a),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(points_a)
        )

        current_a = points_a[
            start:end
        ]

        diff = (
            current_a[:, None, :]
            -
            points_b[None, :, :]
        )

        distance_sq = np.sum(
            diff * diff,
            axis=2
        )

        local_flat_index = np.argmin(
            distance_sq
        )

        local_index = np.unravel_index(
            local_flat_index,
            distance_sq.shape
        )

        local_distance_sq = (
            distance_sq[
                local_index
            ]
        )

        if local_distance_sq < best_distance_sq:

            best_distance_sq = (
                float(local_distance_sq)
            )

            best_a = current_a[
                local_index[0]
            ].copy()

            best_b = points_b[
                local_index[1]
            ].copy()

    if best_a is None or best_b is None:
        raise RuntimeError(
            "failed to find bridge points"
        )

    return (
        best_a,
        best_b,
        float(np.sqrt(best_distance_sq))
    )


# 查找 outer contour / hole contours
def find_outer_and_holes(
        binary_mask
):

    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None:
        return (
            contours,
            [],
            []
        )

    hierarchy = hierarchy[0]

    outer_indices = []
    hole_indices = []

    for index, h in enumerate(
        hierarchy
    ):

        parent = int(h[3])

        if parent == -1:

            outer_indices.append(
                index
            )

        else:

            hole_indices.append(
                index
            )

    return (
        contours,
        outer_indices,
        hole_indices
    )


# 在 hole 和 outer 之间切一条窄缝
def create_bridge_cut(
        mask,
        hole_contour,
        outer_contour,
        bridge_width=2,
        extension=4
):

    hole_point, outer_point, distance = (
        find_closest_points(
            hole_contour,
            outer_contour,
            MAX_BRIDGE_SAMPLE_POINTS
        )
    )

    vector = (
        outer_point -
        hole_point
    )

    length = float(
        np.linalg.norm(vector)
    )

    if length < 1e-6:
        return mask

    direction = (
        vector /
        length
    )

    start_point = (
        hole_point -
        direction *
        extension
    )

    end_point = (
        outer_point +
        direction *
        extension
    )

    start_point = tuple(
        np.round(
            start_point
        ).astype(np.int32)
    )

    end_point = tuple(
        np.round(
            end_point
        ).astype(np.int32)
    )
    # 删除前景，产生 slit
    cv2.line(
        mask,
        start_point,
        end_point,
        color=0,
        thickness=bridge_width,
        lineType=cv2.LINE_8
    )

    return mask


# 将带 hole 的组件转换成单一 polygon mask
def bridge_holes_to_single_polygon_mask(
        component_mask,
        bridge_width=2,
        extension=4
):

    bridged_mask = (
        component_mask.copy()
    )
    # 第一次获取 outer / holes
    (
        contours,
        outer_indices,
        hole_indices
    ) = find_outer_and_holes(
        bridged_mask
    )

    if len(outer_indices) == 0:
        return (
            bridged_mask,
            0
        )

    # 选择面积最大的 outer contour
    outer_idx = max(
        outer_indices,
        key=lambda i: abs(
            cv2.contourArea(
                contours[i]
            )
        )
    )

    outer_contour = contours[
        outer_idx
    ]

    # 记录真正需要处理的 hole
    valid_holes = []

    for hole_idx in hole_indices:

        hole_contour = contours[
            hole_idx
        ]

        hole_area = abs(
            cv2.contourArea(
                hole_contour
            )
        )

        if hole_area >= MIN_HOLE_AREA:

            valid_holes.append(
                hole_contour
            )
    if len(valid_holes) == 0:

        return (
            bridged_mask,
            0
        )
    # 对每个 hole 开 slit
    for hole_contour in valid_holes:

        bridged_mask = (
            create_bridge_cut(
                bridged_mask,
                hole_contour,
                outer_contour,
                bridge_width=bridge_width,
                extension=extension
            )
        )

    # 验证是否还存在 hole
    (
        check_contours,
        check_outer_indices,
        check_hole_indices
    ) = find_outer_and_holes(
        bridged_mask
    )

    remaining_holes = len(
        check_hole_indices
    )

    if remaining_holes > 0:

        # 第二次增加宽度
        retry_width = max(
            bridge_width + 1,
            3
        )

        for hole_contour in valid_holes:

            bridged_mask = (
                create_bridge_cut(
                    bridged_mask,
                    hole_contour,
                    outer_contour,
                    bridge_width=retry_width,
                    extension=extension + 2
                )
            )

        (
            check_contours,
            check_outer_indices,
            check_hole_indices
        ) = find_outer_and_holes(
            bridged_mask
        )

        remaining_holes = len(
            check_hole_indices
        )

    # 最终只取 external contour 对应区域
    external_contours, _ = cv2.findContours(
        bridged_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(external_contours) == 0:

        raise RuntimeError(
            "failed to obtain bridged external contour"
        )

    # 选择面积最大的 contour
    largest_idx = max(
        range(len(external_contours)),
        key=lambda i: abs(
            cv2.contourArea(
                external_contours[i]
            )
        )
    )

    final_mask = np.zeros_like(
        bridged_mask
    )

    cv2.drawContours(
        final_mask,
        external_contours,
        largest_idx,
        255,
        thickness=cv2.FILLED
    )

    return (
        final_mask,
        len(valid_holes)
    )


def contour_to_polygon(
        contour,
        epsilon_ratio=0.001,
        max_epsilon_px=None
):
    """
    contour -> LabelMe polygon points
    """

    if contour is None:
        return None

    if len(contour) < 3:
        return None

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:
        return None

    epsilon = (
        epsilon_ratio *
        perimeter
    )

    if max_epsilon_px is not None:

        epsilon = min(
            epsilon,
            max_epsilon_px
        )

    polygon = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    if polygon is None:
        return None

    if len(polygon) < 3:
        return None

    points = (
        polygon
        .reshape(-1, 2)
        .astype(np.float64)
        .tolist()
    )

    return points


# 构造 LabelMe Polygon
def build_polygon_shape(
        label,
        points,
        group_id=None,
        description=""
):
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": description,
        "shape_type": "polygon",
        "flags": {}
    }


# mask 转LabelMe JSON
def mask_to_labelme(
        mask_path,
        image_path,
        output_json,
        min_area=50,
        epsilon_ratio=0.003,
        bridge_width=2,
        bridge_extension=4
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

    image_height, image_width = (
        image.shape[:2]
    )
    # 尺寸检查
    if mask.shape[:2] != (
        image_height,
        image_width
    ):

        raise RuntimeError(
            "mask/image size mismatch: "
            f"mask={mask.shape[:2]}, "
            f"image={(image_height, image_width)}"
        )

    try:

        image_rel_path = os.path.relpath(
            image_path,
            start=output_json.parent
        ).replace("\\", "/")

    except ValueError:

        image_rel_path = (
            str(image_path)
            .replace("\\", "/")
        )

    labelme_data = {

        "version": "5.5.0",

        "flags": {},

        "shapes": [],

        "imagePath": image_rel_path,

        "imageData": None,

        "imageHeight": int(
            image_height
        ),

        "imageWidth": int(
            image_width
        )
    }
    total_shapes = 0

    normal_polygon_shapes = 0

    bridged_polygon_shapes = 0

    detected_holes = 0

    skipped_small = 0

    skipped_invalid = 0
    class_ids = np.unique(
        mask
    )

    for cls_id in class_ids:

        cls_id = int(
            cls_id
        )

        if cls_id == 0:
            continue

        label = CLASS_MAP.get(
            cls_id,
            str(cls_id)
        )

        binary = (
            mask == cls_id
        ).astype(np.uint8)

        (
            num_components,
            component_labels,
            stats,
            _
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        for component_id in range(
            1,
            num_components
        ):

            pixel_area = int(
                stats[
                    component_id,
                    cv2.CC_STAT_AREA
                ]
            )

            if pixel_area < min_area:

                skipped_small += 1

                continue

            x = int(
                stats[
                    component_id,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    component_id,
                    cv2.CC_STAT_TOP
                ]
            )

            width = int(
                stats[
                    component_id,
                    cv2.CC_STAT_WIDTH
                ]
            )

            height = int(
                stats[
                    component_id,
                    cv2.CC_STAT_HEIGHT
                ]
            )
            component_mask = (
                component_labels[
                    y:y + height,
                    x:x + width
                ] == component_id
            ).astype(np.uint8)

            component_mask_255 = (
                component_mask * 255
            ).astype(np.uint8)

            (
                contours,
                outer_indices,
                hole_indices
            ) = find_outer_and_holes(
                component_mask_255
            )

            hole_count = 0

            valid_hole_indices = []

            for hole_idx in hole_indices:

                hole_area = abs(
                    cv2.contourArea(
                        contours[hole_idx]
                    )
                )

                if hole_area >= MIN_HOLE_AREA:

                    valid_hole_indices.append(
                        hole_idx
                    )

            hole_count = len(
                valid_hole_indices
            )

            if hole_count > 0:

                detected_holes += hole_count

            if len(outer_indices) == 0:

                skipped_invalid += 1

                continue

            if hole_count == 0:

                outer_idx = max(
                    outer_indices,
                    key=lambda i: abs(
                        cv2.contourArea(
                            contours[i]
                        )
                    )
                )

                contour = contours[
                    outer_idx
                ]

                contour_area = abs(
                    cv2.contourArea(
                        contour
                    )
                )

                if contour_area < min_area:

                    skipped_small += 1

                    continue

                points = contour_to_polygon(
                    contour,
                    epsilon_ratio=epsilon_ratio
                )

                if points is None:

                    skipped_invalid += 1

                    continue

                for point in points:

                    point[0] += x

                    point[1] += y

                shape = build_polygon_shape(
                    label=label,
                    points=points,
                    description="polygon"
                )

                labelme_data[
                    "shapes"
                ].append(shape)

                total_shapes += 1

                normal_polygon_shapes += 1

            else:

                (
                    bridged_mask,
                    actual_hole_count
                ) = bridge_holes_to_single_polygon_mask(
                    component_mask_255,
                    bridge_width=bridge_width,
                    extension=bridge_extension
                )

                final_contours, _ = cv2.findContours(
                    bridged_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )

                if len(final_contours) == 0:

                    skipped_invalid += 1

                    continue

                final_contour = max(
                    final_contours,
                    key=lambda c: abs(
                        cv2.contourArea(c)
                    )
                )

                final_area = abs(
                    cv2.contourArea(
                        final_contour
                    )
                )

                if final_area < min_area:

                    skipped_small += 1

                    continue

                points = contour_to_polygon(
                    final_contour,

                    epsilon_ratio=epsilon_ratio,

                    max_epsilon_px=(
                        BRIDGED_MAX_EPSILON_PX
                    )
                )

                if points is None:

                    skipped_invalid += 1

                    continue

                for point in points:

                    point[0] += x

                    point[1] += y

                shape = build_polygon_shape(
                    label=label,
                    points=points,
                    description=(
                        "bridged_polygon"
                    )
                )

                labelme_data[
                    "shapes"
                ].append(shape)

                total_shapes += 1

                bridged_polygon_shapes += 1

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

    return {
        "shapes": total_shapes,
        "normal_polygon_shapes": (
            normal_polygon_shapes
        ),
        "bridged_polygon_shapes": (
            bridged_polygon_shapes
        ),
        "holes": detected_holes,
        "skipped_small": skipped_small,
        "skipped_invalid": skipped_invalid
    }


# 批量转换
def batch_convert(
        mask_dir,
        image_dir,
        json_dir,
        min_area=50,
        epsilon_ratio=0.003,
        num_workers=1,
        bridge_width=2,
        bridge_extension=4
):
    """
    批量 mask -> LabelMe JSON
    """

    mask_dir = Path(
        mask_dir
    )

    image_dir = Path(
        image_dir
    )

    json_dir = Path(
        json_dir
    )

    json_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    image_map = {}

    for img_path in image_dir.iterdir():

        if (
            img_path.is_file()
            and
            img_path.suffix.lower()
            in IMAGE_SUFFIX
        ):

            image_map[
                img_path.stem
            ] = img_path

    mask_files = sorted(
        mask_dir.glob("*.png")
    )

    print(
        f"find mask: {len(mask_files)}"
    )

    tasks = []

    skipped = 0

    for mask_path in mask_files:

        stem = mask_path.stem

        image_path = image_map.get(
            stem
        )

        if image_path is None:

            print(
                f"[skip] image not found: "
                f"{stem}"
            )

            skipped += 1

            continue

        output_json = (
            json_dir /
            f"{stem}.json"
        )

        tasks.append(
            (
                mask_path,
                image_path,
                output_json
            )
        )

    success = 0
    failed = 0

    total_shapes = 0
    total_normal_polygons = 0
    total_bridged_polygons = 0
    total_holes = 0
    total_skipped_small = 0
    total_skipped_invalid = 0
    if num_workers <= 1:

        with tqdm(
            tasks,
            desc="Converting",
            unit="file"
        ) as pbar:

            for (
                mask_path,
                image_path,
                output_json
            ) in pbar:

                stem = mask_path.stem

                try:

                    result = mask_to_labelme(
                        mask_path=mask_path,
                        image_path=image_path,
                        output_json=output_json,
                        min_area=min_area,
                        epsilon_ratio=epsilon_ratio,
                        bridge_width=bridge_width,
                        bridge_extension=bridge_extension
                    )

                    success += 1

                    total_shapes += result[
                        "shapes"
                    ]

                    total_normal_polygons += result[
                        "normal_polygon_shapes"
                    ]

                    total_bridged_polygons += result[
                        "bridged_polygon_shapes"
                    ]

                    total_holes += result[
                        "holes"
                    ]

                    total_skipped_small += result[
                        "skipped_small"
                    ]

                    total_skipped_invalid += result[
                        "skipped_invalid"
                    ]

                except Exception as e:

                    failed += 1

                    tqdm.write(
                        f"[fail] "
                        f"{stem}: {e}"
                    )

                pbar.set_postfix(
                    success=success,
                    failed=failed
                )

    else:

        from concurrent.futures import (
            ThreadPoolExecutor,
            as_completed
        )

        with ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:

            future_to_stem = {}

            for (
                mask_path,
                image_path,
                output_json
            ) in tasks:

                stem = mask_path.stem

                future = executor.submit(
                    mask_to_labelme,
                    mask_path,
                    image_path,
                    output_json,
                    min_area,
                    epsilon_ratio,
                    bridge_width,
                    bridge_extension
                )

                future_to_stem[
                    future
                ] = stem

            with tqdm(
                total=len(tasks),
                desc="Converting",
                unit="file"
            ) as pbar:

                for future in as_completed(
                    future_to_stem
                ):

                    stem = future_to_stem[
                        future
                    ]

                    try:

                        result = future.result()

                        success += 1

                        total_shapes += result[
                            "shapes"
                        ]

                        total_normal_polygons += result[
                            "normal_polygon_shapes"
                        ]

                        total_bridged_polygons += result[
                            "bridged_polygon_shapes"
                        ]

                        total_holes += result[
                            "holes"
                        ]

                        total_skipped_small += result[
                            "skipped_small"
                        ]

                        total_skipped_invalid += result[
                            "skipped_invalid"
                        ]

                    except Exception as e:

                        failed += 1

                        tqdm.write(
                            f"[fail] "
                            f"{stem}: {e}"
                        )

                    pbar.update(1)

                    pbar.set_postfix(
                        success=success,
                        failed=failed
                    )
if __name__ == "__main__":

    batch_convert(

        mask_dir=(
            r"D:\PythonProject\environment_segment\output\masks"
        ),

        image_dir=(
            r"D:\PythonProject\My_Mask2Former\test_images\images"
        ),

        json_dir=(
            r"D:\PythonProject\environment_segment\output\jsons"
        ),

        min_area=100,

        epsilon_ratio=0.0001,

        num_workers=16,

        # ----------------------------------------------------
        # hole -> 单 polygon 的桥接参数
        # ----------------------------------------------------

        bridge_width=2,

        bridge_extension=4
    )
