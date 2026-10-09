import json
import os
import shutil


# ============================================================
# 配置
# ============================================================

JSON_DIR = r"D:\DataSets\Cable\jsons"
IMAGE_DIR = r"D:\DataSets\Cable\images"

LABEL_MAP = {
    "conductor": "conductor_wire",
    # "white_cable": "conductor",
}


# ============================================================
# 查找图片
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
]


def find_image(image_dir, image_name):

    # 首先尝试原始文件名
    path = os.path.join(image_dir, image_name)

    if os.path.isfile(path):
        return path

    # 根据 stem 搜索不同扩展名
    stem = os.path.splitext(image_name)[0]

    for ext in IMAGE_EXTENSIONS:

        path = os.path.join(
            image_dir,
            stem + ext
        )

        if os.path.isfile(path):
            return path

    return None


# ============================================================
# 修改单个 JSON
# ============================================================

def process_json(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    modified_labels = 0

    # --------------------------------------------------------
    # 1. 修改 label
    # --------------------------------------------------------

    shapes = data.get("shapes", [])

    for shape in shapes:

        if not isinstance(shape, dict):
            continue

        old_label = shape.get("label")

        if old_label in LABEL_MAP:

            new_label = LABEL_MAP[old_label]

            shape["label"] = new_label

            modified_labels += 1

            print(
                f"    label: "
                f"{old_label} -> {new_label}"
            )

    # --------------------------------------------------------
    # 2. 修复 imagePath
    # --------------------------------------------------------

    image_path_old = data.get("imagePath")

    if image_path_old:

        # 只取原 imagePath 的文件名
        image_name = os.path.basename(
            image_path_old.replace("\\", "/")
        )

    else:

        # 如果 JSON 没有 imagePath
        # 使用 JSON 文件名寻找图片
        image_name = os.path.splitext(
            os.path.basename(json_path)
        )[0]

    # 在真实图片目录中寻找
    real_image_path = find_image(
        IMAGE_DIR,
        image_name
    )

    if real_image_path is None:

        print(
            f"    [警告] 找不到图片: "
            f"{image_name}"
        )

    else:

        # 计算从 JSON 目录到图片目录的相对路径
        relative_image_path = os.path.relpath(
            real_image_path,
            os.path.dirname(json_path)
        )

        # LabelMe 推荐使用 /
        relative_image_path = relative_image_path.replace(
            "\\",
            "/"
        )

        data["imagePath"] = relative_image_path

        print(
            f"    imagePath: "
            f"{image_path_old} -> "
            f"{relative_image_path}"
        )

        # ----------------------------------------------------
        # 关键：
        # imageData 设置为 None 没问题，
        # 但 imagePath 必须能够找到真实图片
        # ----------------------------------------------------

    # --------------------------------------------------------
    # 3. 保存
    # --------------------------------------------------------

    # 先备份
    backup_path = json_path + ".bak"

    if not os.path.exists(backup_path):

        shutil.copy2(
            json_path,
            backup_path
        )

    # 临时文件
    temp_path = json_path + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # 替换原文件
    os.replace(
        temp_path,
        json_path
    )

    return modified_labels


# ============================================================
# 批量处理
# ============================================================

def main():

    json_files = [
        f
        for f in os.listdir(JSON_DIR)
        if f.lower().endswith(".json")
    ]

    print(
        f"发现 {len(json_files)} 个 JSON 文件"
    )

    print("=" * 70)

    total_modified = 0

    for file_name in json_files:

        json_path = os.path.join(
            JSON_DIR,
            file_name
        )

        print(
            f"\n处理: {file_name}"
        )

        count = process_json(
            json_path
        )

        total_modified += count

    print("\n")
    print("=" * 70)
    print("处理完成")
    print("=" * 70)

    print(
        f"总 JSON 数量: {len(json_files)}"
    )

    print(
        f"总标签修改数: {total_modified}"
    )


if __name__ == "__main__":
    main()