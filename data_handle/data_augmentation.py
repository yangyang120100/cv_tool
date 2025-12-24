import os
import cv2
from pathlib import Path
import albumentations as A

# =====================================================
# 1. 参数设置
# =====================================================

AUG_TIMES = 3  # 每张图增强几次

IMAGE_DIR = r"D:\DataBase\line_train_dataset\val\images"
MASK_DIR = r"D:\DataBase\line_train_dataset\val\masks"

OUT_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "augmented_images")
OUT_MASK_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "augmented_masks")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =====================================================
# 2. 数据增强策略（无 Resize、无预处理）
# =====================================================

transform = A.Compose([
    # ---------- 几何变换 ----------
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=0.5
    ),

    # ---------- 颜色增强（仅 image） ----------
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.2),
])

# =====================================================
# 3. 读取 image & mask（不做任何处理）
# =====================================================

def load_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)  # BGR
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    return image, mask

# =====================================================
# 4. 保存 image & mask（原样保存）
# =====================================================

def save_image_mask(image, mask, name):
    # cv2.imshow("image", image)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(OUT_IMAGE_DIR, name), image)
    mask_name=Path(name).stem
    cv2.imwrite(os.path.join(OUT_MASK_DIR, mask_name+".png"), mask)

# =====================================================
# 5. 执行增强
# =====================================================

image_names = sorted(os.listdir(IMAGE_DIR))

for idx, name in enumerate(image_names):
    img_path = os.path.join(IMAGE_DIR, name)
    mask_name=Path(name).stem
    mask_path = os.path.join(MASK_DIR, mask_name+".png")

    image, mask = load_image_mask(img_path, mask_path)

    if not os.path.exists(mask_path):
        print(f"⚠ mask 不存在: {name}")
        continue

    image, mask = load_image_mask(img_path, mask_path)

    # ---------- 保存原图 ----------
    save_image_mask(image, mask, name)

    # ---------- 保存增强图 ----------
    for i in range(AUG_TIMES):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        new_name = name.replace(".", f"_aug{i}.")

        save_image_mask(aug_image, aug_mask, new_name)

    print(f"[{idx + 1}/{len(image_names)}] processed {name}")

print("数据增强完成 ✅")
