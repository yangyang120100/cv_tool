import os
import cv2
import hashlib
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import albumentations as A

"""
工业级语义分割数据增强脚本
兼容 Albumentations 新版 API
使用 argparse 进行参数配置
"""

# =====================================================
# 1 全局变量（将在 main 中根据 args 重新赋值）
# =====================================================
AUG_TIMES = None
NUM_WORKERS = None
IMAGE_DIR = None
MASK_DIR = None
OUT_IMAGE_DIR = None
OUT_MASK_DIR = None

VALID_EXT = {".jpg", ".jpeg", ".png"}

# =====================================================
# 2 数据增强策略
# =====================================================
transform = A.Compose([

    # 必定发生一种几何变换
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
    ], p=1.0),

    A.Affine(
        translate_percent=(0.0, 0.08),
        scale=(0.9, 1.15),
        rotate=(-20, 20),
        shear=(-5, 5),
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
        p=0.7
    ),

    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.GaussNoise(),
    ], p=0.3),

    A.RandomBrightnessContrast(p=0.4),

])

# =====================================================
# 3 工具函数
# =====================================================
def img_hash(img):
    """计算图像hash用于去重"""
    return hashlib.md5(img.tobytes()).hexdigest()


def load_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask


def save_image_mask(image, mask, name):
    cv2.imwrite(os.path.join(OUT_IMAGE_DIR, name), image)
    mask_name = Path(name).stem + ".png"
    cv2.imwrite(os.path.join(OUT_MASK_DIR, mask_name), mask)


# =====================================================
# 4 单样本处理
# =====================================================
def process_one(name):
    stem = Path(name).stem

    img_path = os.path.join(IMAGE_DIR, name)
    mask_path = os.path.join(MASK_DIR, stem + ".png")

    if not os.path.exists(mask_path):
        return

    image, mask = load_image_mask(img_path, mask_path)

    if image is None or mask is None:
        return

    # 保存原图
    save_image_mask(image, mask, name)

    origin_hash = img_hash(image)
    used_hash = set()

    for i in range(AUG_TIMES):
        for _ in range(10):  # 最多尝试10次
            augmented = transform(image=image, mask=mask)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            h = img_hash(aug_img)

            # 与原图相同
            if h == origin_hash:
                continue
            # 与之前增强重复
            if h in used_hash:
                continue

            used_hash.add(h)

            new_name = f"{stem}_aug{i}.jpg"
            save_image_mask(aug_img, aug_mask, new_name)
            break


# =====================================================
# 5 参数解析
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(description="语义分割数据增强脚本")
    parser.add_argument('--image_dir', '-i', required=False,
                        default=r"D:\DataBase\cabel_train_datas\images",
                        help='输入图像目录')
    parser.add_argument('--mask_dir', '-m', required=False,
                        default=r"D:\DataBase\cabel_train_datas\masks",
                        help='输入掩码目录')
    parser.add_argument('--aug_times', type=int, default=4,
                        help='每张图像的增强次数 (默认: 4)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='并发线程数 (默认: 32)')
    # 输出目录自动生成，无需额外参数
    return parser.parse_args()


# =====================================================
# 6 主程序
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    # 将参数赋值给全局变量，供各函数使用
    AUG_TIMES = args.aug_times
    NUM_WORKERS = args.num_workers
    IMAGE_DIR = args.image_dir
    MASK_DIR = args.mask_dir

    # 自动生成输出目录（位于输入目录的父目录下，加 "aug_" 前缀）
    OUT_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), f"aug_{Path(IMAGE_DIR).stem}")
    OUT_MASK_DIR = os.path.join(os.path.dirname(IMAGE_DIR), f"aug_{Path(MASK_DIR).stem}")

    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    # 获取所有图像文件
    image_names = [
        f for f in os.listdir(IMAGE_DIR)
        if Path(f).suffix.lower() in VALID_EXT
    ]

    print("\n🚀 开始数据增强")
    print(f"图片数量: {len(image_names)}")
    print(f"增强倍数: {AUG_TIMES}")
    print(f"线程数: {NUM_WORKERS}")
    print(f"输出图像目录: {OUT_IMAGE_DIR}")
    print(f"输出掩码目录: {OUT_MASK_DIR}\n")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(
            executor.map(process_one, image_names),
            total=len(image_names)
        ))

    print("\n🎉 数据增强完成")