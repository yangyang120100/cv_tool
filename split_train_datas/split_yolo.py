import os
import random
import shutil
from pathlib import Path


# -------------------------------------------------
# 工具函数：查找标签文件（txt 优先，其次 json）
# -------------------------------------------------
def find_label_file(labels_dir, stem):
    txt_path = Path(labels_dir) / f"{stem}.txt"
    json_path = Path(labels_dir) / f"{stem}.json"

    if txt_path.exists():
        return txt_path
    if json_path.exists():
        return json_path
    return None


# -------------------------------------------------
# 数据集分割主函数
# -------------------------------------------------
def split_yolo_dataset(
    images_dir,
    labels_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)

    output_path = Path(output_dir)
    splits = ['train', 'val', 'test']

    # 创建目录结构
    for split in splits:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 搜索所有图像文件（去重）
    # -------------------------------------------------
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    seen = set()

    for ext in image_extensions:
        for p in Path(images_dir).glob(f"*{ext}"):
            if p.name not in seen:
                image_files.append(p)
                seen.add(p.name)
        for p in Path(images_dir).glob(f"*{ext.upper()}"):
            if p.name not in seen:
                image_files.append(p)
                seen.add(p.name)

    print(f"找到图像文件: {len(image_files)}")

    # -------------------------------------------------
    # 检查标签（txt / json）
    # -------------------------------------------------
    valid_images = []
    missing_labels = []

    for img in image_files:
        label = find_label_file(labels_dir, img.stem)
        if label:
            valid_images.append(img)
        else:
            missing_labels.append(img.stem)

    if missing_labels:
        print(f"⚠️ 缺失标签的图像数量: {len(missing_labels)}")
        print("示例:", missing_labels[:10])

    image_files = valid_images
    print(f"有效图像-标签对: {len(image_files)}")

    if not image_files:
        print("❌ 没有可用数据")
        return

    # -------------------------------------------------
    # 打乱 & 切分
    # -------------------------------------------------
    random.shuffle(image_files)

    total = len(image_files)
    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)

    splits_files = {
        "train": image_files[:train_n],
        "val": image_files[train_n:train_n + val_n],
        "test": image_files[train_n + val_n:]
    }

    print("\n数据集划分:")
    for k, v in splits_files.items():
        print(f"{k}: {len(v)}")

    # -------------------------------------------------
    # 复制文件
    # -------------------------------------------------
    for split, files in splits_files.items():
        print(f"\n处理 {split} 集...")

        for img_path in files:
            label_path = find_label_file(labels_dir, img_path.stem)

            shutil.copy2(
                img_path,
                output_path / split / "images" / img_path.name
            )

            shutil.copy2(
                label_path,
                output_path / split / "labels" / label_path.name
            )

        print(f"{split} 集完成")

    # -------------------------------------------------
    # 生成 YAML
    # -------------------------------------------------
    create_yaml_file(output_path, splits_files)

    print("\n✅ 数据集分割完成")


# -------------------------------------------------
# 生成 data.yaml
# -------------------------------------------------
def create_yaml_file(output_path, splits_files):
    nc = detect_num_classes(output_path / "train" / "labels")

    yaml_content = f"""
# YOLO Dataset Config
path: {output_path.resolve()}

train: train/images
val: val/images
test: test/images

nc: {nc}
names: {get_class_names(nc)}

# statistics
# train: {len(splits_files['train'])}
# val: {len(splits_files['val'])}
# test: {len(splits_files['test'])}
"""

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print(f"生成 data.yaml: {yaml_path}")


# -------------------------------------------------
# 从 YOLO txt 标签检测类别数
# -------------------------------------------------
def detect_num_classes(labels_dir):
    class_ids = set()

    for txt in Path(labels_dir).glob("*.txt"):
        with open(txt, "r") as f:
            for line in f:
                if line.strip():
                    class_ids.add(int(line.split()[0]))

    if class_ids:
        print(f"检测到类别 ID: {sorted(class_ids)}")
        return max(class_ids) + 1

    print("⚠️ 未检测到 txt 标签，默认 nc=1")
    return 1


def get_class_names(nc):
    if nc == 1:
        return ["object"]
    return [f"class_{i}" for i in range(nc)]


# -------------------------------------------------
# 数据完整性校验（支持 txt + json）
# -------------------------------------------------
def verify_dataset(output_dir):
    print("\n验证数据集完整性...")

    for split in ["train", "val", "test"]:
        img_dir = Path(output_dir) / split / "images"
        lbl_dir = Path(output_dir) / split / "labels"

        images = {p.stem for p in img_dir.glob("*")}
        labels = {
            p.stem for p in lbl_dir.glob("*.txt")
        } | {
            p.stem for p in lbl_dir.glob("*.json")
        }

        print(f"\n{split} 集:")
        print(f"images: {len(images)}, labels: {len(labels)}")

        miss = images - labels
        extra = labels - images

        if miss:
            print(f"⚠️ 缺失标签: {list(miss)[:5]}")
        if extra:
            print(f"⚠️ 多余标签: {list(extra)[:5]}")


# -------------------------------------------------
# main
# -------------------------------------------------
if __name__ == "__main__":
    images_directory = r"D:\Projects\U-2-Net\make_train_datas\images"
    labels_directory = r"D:\Projects\U-2-Net\make_train_datas\labels"
    output_directory = r"D:\Projects\Scripting_tool\output"

    split_yolo_dataset(
        images_dir=images_directory,
        labels_dir=labels_directory,
        output_dir=output_directory,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    verify_dataset(output_directory)

    print("\n使用说明:")
    print("1. 如果使用 YOLO 训练，请确保 labels 中存在 .txt")
    print("2. json 标签会被原样保留（用于追溯/转换）")
    print("3. 训练命令: yolo train data=data.yaml ...")
