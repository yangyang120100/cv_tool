import cv2
import json
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


# ============================================================
# 类别定义
# ============================================================

ALL_CLASSES = [
    "background",
    "conductor_wire",
    "lead_wire"
]

# ALL_CLASSES =['background', 'asphalt-road', 'concrete-pavement', 'brick-road', 'dirt', 'gravel', 'inland-water', 'standing-water', 'woods', 'tall-vegetation', 'low-vegetation', 'roof', 'wall', 'fence', 'fence-post', 'snow', 'concrete-pole', 'angle-steel-tower', 'steel-tube-tower', 'conductor', 'sky', 'obstacle', 'car', 'bus', 'work-vehicle', 'large-vehicle', 'bicycle', 'person', 'manhole-cover', 'distant-building', 'traffic-light', 'house', 'tricycle']

CLASS_MAP = {
    i: j for i, j in enumerate(ALL_CLASSES)
}


# ============================================================
# 支持的图像格式
# ============================================================

IMAGE_SUFFIX = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# Hole -> 单 Polygon 桥接参数
# ============================================================

# 桥接缝宽度，单位：像素
#
# 推荐：
#   1 = 最小改动
#   2 = 更稳定
#
BRIDGE_WIDTH = 2


# 从 hole / outer boundary 向外延伸多少像素
#
# 作用：
# 防止由于 OpenCV 离散像素导致：
#     hole --- cut --- outer
# 没有真正贯通
#
BRIDGE_EXTENSION = 4


# 带 hole 的 polygon 在近似时允许的最大 epsilon
#
# 普通 polygon：
#     epsilon = epsilon_ratio * perimeter
#
# 带 hole 的桥接 polygon：
#     epsilon = min(
#         epsilon_ratio * perimeter,
#         BRIDGED_MAX_EPSILON_PX
#     )
#
# 不能太大，否则窄桥会被 approxPolyDP 删除。
#
BRIDGED_MAX_EPSILON_PX = 0.75


# 寻找最近桥接点时最多采样多少个 contour 点
#
# 可以避免 contour 非常大时进行巨量距离计算。
#
MAX_BRIDGE_SAMPLE_POINTS = 2000


# 小于这个面积的 hole 不单独进行桥接
#
# 设为 1 表示理论上所有 hole 都处理。
#
MIN_HOLE_AREA = 1.0


# ============================================================
# contour 采样
# ============================================================

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


# ============================================================
# 寻找两个 contour 的最近点
# ============================================================

def find_closest_points(
        contour_a,
        contour_b,
        max_sample_points=2000,
        chunk_size=256
):
    """
    在两个 contour 之间寻找近似欧氏距离最近的点。

    为避免一次性生成超大的 NxM 距离矩阵，
    使用 chunk 计算。

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

        # ----------------------------------------------------
        # current_a:
        #     [Na, 2]
        #
        # points_b:
        #     [Nb, 2]
        #
        # diff:
        #     [Na, Nb, 2]
        # ----------------------------------------------------

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


# ============================================================
# 查找 outer contour / hole contours
# ============================================================

def find_outer_and_holes(
        binary_mask
):
    """
    使用 RETR_CCOMP：

        parent == -1
            -> outer contour

        parent != -1
            -> hole contour
    """

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


# ============================================================
# 在 hole 和 outer 之间切一条窄缝
# ============================================================

def create_bridge_cut(
        mask,
        hole_contour,
        outer_contour,
        bridge_width=2,
        extension=4
):
    """
    从 hole 向 outer boundary 开一条窄缝。

    注意：

    我们不是“画前景桥”，而是：

        将前景区域删除

    即：

        foreground -> 0

    这样原来的 hole 就通过这条缝
    与外部 background 相连。

    最终重新提取外轮廓后，
    就只剩下一个 polygon。

    """

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

    # --------------------------------------------------------
    # hole 这一端：
    #
    # 向 hole 内部延伸
    #
    # --------------------------------------------------------

    start_point = (
        hole_point -
        direction *
        extension
    )

    # --------------------------------------------------------
    # outer 这一端：
    #
    # 穿过 outer boundary，
    # 延伸到外部
    #
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 删除前景，产生 slit
    # --------------------------------------------------------

    cv2.line(
        mask,
        start_point,
        end_point,
        color=0,
        thickness=bridge_width,
        lineType=cv2.LINE_8
    )

    return mask


# ============================================================
# 将带 hole 的组件转换成单一 polygon mask
# ============================================================

def bridge_holes_to_single_polygon_mask(
        component_mask,
        bridge_width=2,
        extension=4
):
    """
    输入：

        一个连通组件 binary mask

    输出：

        不含 hole 的 binary mask

    原理：

        hole
          ↓
        cut/slit
          ↓
        与外部背景连通
          ↓
        RETR_EXTERNAL
          ↓
        单个 polygon
    """

    bridged_mask = (
        component_mask.copy()
    )

    # --------------------------------------------------------
    # 第一次获取 outer / holes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 选择面积最大的 outer contour
    #
    # 对一个 connected component 理论上只有一个 outer，
    # 这里仍然做保护。
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 记录真正需要处理的 hole
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 没有 hole
    # --------------------------------------------------------

    if len(valid_holes) == 0:

        return (
            bridged_mask,
            0
        )

    # --------------------------------------------------------
    # 对每个 hole 开 slit
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 验证是否还存在 hole
    #
    # 某些极端离散情况可能一次没有完全打通。
    # 如果仍存在 hole，增加 bridge 宽度再处理一次。
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 最终只取 external contour 对应区域
    #
    # 由于 hole 已经被 slit 打开，
    # RETR_EXTERNAL 应该只剩一个外部区域。
    # --------------------------------------------------------

    external_contours, _ = cv2.findContours(
        bridged_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(external_contours) == 0:

        raise RuntimeError(
            "failed to obtain bridged external contour"
        )

    # --------------------------------------------------------
    # 选择面积最大的 contour
    # --------------------------------------------------------

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


# ============================================================
# contour -> polygon points
# ============================================================

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


# ============================================================
# 构造 LabelMe Polygon
# ============================================================

def build_polygon_shape(
        label,
        points,
        group_id=None,
        description=""
):
    """
    标准 LabelMe polygon。
    """

    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": description,
        "shape_type": "polygon",
        "flags": {}
    }


# ============================================================
# 单张 mask -> LabelMe JSON
# ============================================================

def mask_to_labelme(
        mask_path,
        image_path,
        output_json,
        min_area=50,
        epsilon_ratio=0.003,
        bridge_width=2,
        bridge_extension=4
):
    """
    将语义分割 mask 转换为 LabelMe JSON。

    规则：

    --------------------------------------------------------
    没有 hole
        -> 一个普通 polygon

    有 hole
        -> 对 hole 开窄缝
        -> 转换为一个普通 polygon

    最终：
        所有 shape 都是 shape_type="polygon"
        所有目标都有可移动顶点
    --------------------------------------------------------
    """

    # ========================================================
    # 读取 mask
    # ========================================================

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise RuntimeError(
            f"cannot read mask: {mask_path}"
        )

    # ========================================================
    # 读取原始图像
    # ========================================================

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

    # ========================================================
    # 尺寸检查
    # ========================================================

    if mask.shape[:2] != (
        image_height,
        image_width
    ):

        raise RuntimeError(
            "mask/image size mismatch: "
            f"mask={mask.shape[:2]}, "
            f"image={(image_height, image_width)}"
        )

    # ========================================================
    # imagePath
    # ========================================================

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

    # ========================================================
    # LabelMe JSON
    # ========================================================

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

    # ========================================================
    # 统计
    # ========================================================

    total_shapes = 0

    normal_polygon_shapes = 0

    bridged_polygon_shapes = 0

    detected_holes = 0

    skipped_small = 0

    skipped_invalid = 0

    # ========================================================
    # 遍历类别
    # ========================================================

    class_ids = np.unique(
        mask
    )

    for cls_id in class_ids:

        cls_id = int(
            cls_id
        )

        # ----------------------------------------------------
        # background
        # ----------------------------------------------------

        if cls_id == 0:
            continue

        label = CLASS_MAP.get(
            cls_id,
            str(cls_id)
        )

        # ----------------------------------------------------
        # 当前类别 mask
        # ----------------------------------------------------

        binary = (
            mask == cls_id
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Connected Components
        # ----------------------------------------------------

        (
            num_components,
            component_labels,
            stats,
            _
        ) = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        # ====================================================
        # 遍历每个连通组件
        # ====================================================

        for component_id in range(
            1,
            num_components
        ):

            # ------------------------------------------------
            # 像素面积
            # ------------------------------------------------

            pixel_area = int(
                stats[
                    component_id,
                    cv2.CC_STAT_AREA
                ]
            )

            if pixel_area < min_area:

                skipped_small += 1

                continue

            # ------------------------------------------------
            # ROI
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 提取 component ROI
            # ------------------------------------------------

            component_mask = (
                component_labels[
                    y:y + height,
                    x:x + width
                ] == component_id
            ).astype(np.uint8)

            component_mask_255 = (
                component_mask * 255
            ).astype(np.uint8)

            # ------------------------------------------------
            # 检测 hole
            # ------------------------------------------------

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

            # =================================================
            # 没有 outer
            # =================================================

            if len(outer_indices) == 0:

                skipped_invalid += 1

                continue

            # =================================================
            # 情况 1：
            #
            # 普通无 hole 目标
            # =================================================

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

                # ------------------------------------------------
                # ROI 坐标 -> 原图坐标
                #
                # 这是之前左上角 bug 的关键修复。
                # ------------------------------------------------

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

            # =================================================
            # 情况 2：
            #
            # 有 hole
            #
            # -> 开 slit
            # -> 变成单一 polygon
            # =================================================

            else:

                (
                    bridged_mask,
                    actual_hole_count
                ) = bridge_holes_to_single_polygon_mask(
                    component_mask_255,
                    bridge_width=bridge_width,
                    extension=bridge_extension
                )

                # ------------------------------------------------
                # 对桥接后的 mask 再提取 external contour
                # ------------------------------------------------

                final_contours, _ = cv2.findContours(
                    bridged_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )

                if len(final_contours) == 0:

                    skipped_invalid += 1

                    continue

                # ------------------------------------------------
                # 选择最大 contour
                # ------------------------------------------------

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

                # ------------------------------------------------
                # 非常重要：
                #
                # 对 bridge polygon 限制 epsilon
                #
                # 防止窄 slit 被 approxPolyDP 删除。
                # ------------------------------------------------

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

                # ------------------------------------------------
                # ROI -> 原图
                # ------------------------------------------------

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

    # ========================================================
    # 保存 JSON
    # ========================================================

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

    # ========================================================
    # 返回统计
    # ========================================================

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


# ============================================================
# 批量转换
# ============================================================

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

    # ========================================================
    # 建立 image stem -> path
    # ========================================================

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

    # ========================================================
    # 获取 mask
    # ========================================================

    mask_files = sorted(
        mask_dir.glob("*.png")
    )

    print(
        f"find mask: {len(mask_files)}"
    )

    # ========================================================
    # 创建任务
    # ========================================================

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

    # ========================================================
    # 全局统计
    # ========================================================

    success = 0
    failed = 0

    total_shapes = 0
    total_normal_polygons = 0
    total_bridged_polygons = 0
    total_holes = 0
    total_skipped_small = 0
    total_skipped_invalid = 0

    # ========================================================
    # 单线程
    # ========================================================

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

    # ========================================================
    # 多线程
    # ========================================================

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

    # ========================================================
    # 最终统计
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "Conversion Finished"
    )
    print(
        "=============================================="
    )

    print(
        f"success               : {success}"
    )

    print(
        f"failed                : {failed}"
    )

    print(
        f"skipped               : {skipped}"
    )

    print(
        f"total shapes          : {total_shapes}"
    )

    print(
        f"normal polygons       : {total_normal_polygons}"
    )

    print(
        f"bridged polygons      : {total_bridged_polygons}"
    )

    print(
        f"detected holes        : {total_holes}"
    )

    print(
        f"skipped small         : {total_skipped_small}"
    )

    print(
        f"skipped invalid       : {total_skipped_invalid}"
    )

    print(
        "=============================================="
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    batch_convert(

        mask_dir=(
            r"E:\test_dino\mixed_precision_masks"
        ),

        image_dir=(
            r"E:\test_dino\images"
        ),

        json_dir=(
            r"E:\test_dino\mixed_precision_jsons"
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
