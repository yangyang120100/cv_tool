import os
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A

"""
数据增强脚本
"""



# =====================================================
# 1. 参数设置
# =====================================================

AUG_TIMES = 3
NUM_WORKERS = 8   # ⭐ 建议 = CPU 核数 或 核数*2

IMAGE_DIR = r"D:\DataBase\cabel_train_datas\new_datas\images"
MASK_DIR = r"D:\DataBase\cabel_train_datas\new_datas\masks"

OUT_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), f"augmented_{Path(IMAGE_DIR).stem}")
OUT_MASK_DIR = os.path.join(os.path.dirname(IMAGE_DIR), f"augmented_{Path(MASK_DIR).stem}")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =====================================================
# 2. 数据增强策略（无 Resize、无预处理）
# =====================================================

transform = A.Compose([
    A.HorizontalFlip(p=0.5),                # 以50%的概率水平翻转图片
    A.VerticalFlip(p=0.5),                  # 以50%的概率垂直翻转图片
    A.RandomRotate90(p=0.5),                # 以50%的概率随机旋转90度（0, 90, 180, 270）

    A.ShiftScaleRotate(
        shift_limit=0.05,                   # 平移范围（相对宽高的5%）
        scale_limit=0.1,                    # 缩放范围（±10%）
        rotate_limit=15,                    # 随机旋转角度范围（±15度）
        border_mode=cv2.BORDER_CONSTANT,    # 边界填充方式，使用常数填充
        value=0,                            # 边界填充的像素值（图像）
        mask_value=0,                       # 边界填充的像素值（mask）
        p=0.5                               # 以50%的概率应用该增强
    ),

    A.RandomBrightnessContrast(p=0.4),      # 以40%的概率随机调整亮度和对比度
    A.GaussianBlur(blur_limit=3, p=0.2),    # 以20%的概率进行高斯模糊，最大模糊核为3
])

# =====================================================
# 3. IO 函数
# =====================================================

def load_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask


def save_image_mask(image, mask, name):
    cv2.imwrite(os.path.join(OUT_IMAGE_DIR, name), image)

    mask_name = Path(name).stem + ".png"
    cv2.imwrite(os.path.join(OUT_MASK_DIR, mask_name), mask)

# =====================================================
# 4. 单样本处理函数（线程任务）
# =====================================================

def process_one(name):
    img_path = os.path.join(IMAGE_DIR, name)
    mask_name = Path(name).stem + ".png"
    mask_path = os.path.join(MASK_DIR, mask_name)

    if not os.path.exists(mask_path):
        return f"⚠ mask 不存在: {name}"

    image, mask = load_image_mask(img_path, mask_path)

    if image is None or mask is None:
        return f"❌ 读取失败: {name}"

    # 保存原图
    save_image_mask(image, mask, name)

    # 保存增强图
    for i in range(AUG_TIMES):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        new_name = name.replace(".", f"_aug{i}.")
        save_image_mask(aug_image, aug_mask, new_name)

    return f"✅ processed {name}"

# =====================================================
# 5. 多线程执行
# =====================================================

if __name__ == "__main__":
    image_names = sorted(os.listdir(IMAGE_DIR))
    total = len(image_names)

    print(f"🚀 开始数据增强，共 {total} 张，线程数 = {NUM_WORKERS}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_one, name) for name in image_names]

        for idx, future in enumerate(as_completed(futures), 1):
            print(f"[{idx}/{total}] {future.result()}")

    print("🎉 数据增强完成")
