import os
import glob
import numpy as np
from PIL import Image

# ==================== 配置参数 ====================
IMAGE_DIR = r"E:\test_colmap\images"  # 原始图像文件夹路径
MASK_DIR = r"E:\test_colmap\masks"  # mask 图像文件夹路径
LABELS = [1,2,4]  # 需要保留的标签值列表
OUTPUT_DIR = r"E:\test_colmap\mask_images"  # 输出文件夹路径
BACKGROUND = "transparent"  # 背景处理方式: "transparent", "black", "white"
OUTPUT_EXT = ".png"  # 输出文件扩展名
MASK_EXT = None  # mask 文件扩展名（如 ".png"），None 则自动匹配
# ==================================================

# 支持的图像扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def find_mask_file(mask_dir, base_name, mask_ext=None):
    """查找与图像同名的 mask 文件（扩展名可能不同）"""
    if mask_ext:
        mask_path = os.path.join(mask_dir, base_name + mask_ext)
        if os.path.exists(mask_path):
            return mask_path
        return None

    # 尝试常见扩展名
    common_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']
    for ext in common_exts:
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            return mask_path

    # 模糊匹配任意扩展名
    matches = glob.glob(os.path.join(mask_dir, base_name + '.*'))
    if matches:
        return matches[0]
    return None


def extract_labels_from_mask(image_dir, mask_dir, labels, output_dir,
                             background="transparent", output_ext=".png", mask_ext=None):
    """
    根据 mask 中的指定标签从原始图像中提取对应区域

    参数:
        image_dir: 原始图像文件夹路径
        mask_dir: mask 图像文件夹路径
        labels: 需要保留的标签值列表
        output_dir: 输出文件夹路径
        background: 背景处理方式 ("transparent", "black", "white")
        output_ext: 输出文件扩展名
        mask_ext: mask 文件扩展名（可选）
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 转换标签为集合以便快速查找
    label_set = set(labels)
    print(f"需要保留的标签：{sorted(label_set)}")

    # 统计信息
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 遍历图像文件夹
    for image_filename in sorted(os.listdir(image_dir)):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(image_dir, image_filename)
        base_name = os.path.splitext(image_filename)[0]

        # 查找对应的 mask
        mask_path = find_mask_file(mask_dir, base_name, mask_ext)
        if mask_path is None:
            print(f"⚠️  警告：未找到 {image_filename} 的 mask，跳过")
            skipped_count += 1
            continue

        try:
            # 读取原始图像
            with Image.open(image_path) as img:
                # 复制元数据（EXIF、ICC、DPI 等）
                img_info = img.info.copy()

                # 转换为 RGB（如果必要），再转换为 RGBA
                img_rgb = img.convert('RGB') if img.mode != 'RGB' else img
                img_rgba = img_rgb.convert('RGBA')

                # 读取 mask
                with Image.open(mask_path) as mask_img:
                    # 转换为单通道灰度图像
                    if mask_img.mode != 'L':
                        mask_gray = mask_img.convert('L')
                    else:
                        mask_gray = mask_img

                    # 检查尺寸是否一致
                    if mask_gray.size != img_rgba.size:
                        print(
                            f"⚠️  警告：mask 尺寸 {mask_gray.size} 与图像 {img_rgba.size} 不一致，跳过 {image_filename}")
                        skipped_count += 1
                        continue

                    # 转为 numpy 数组
                    mask_array = np.array(mask_gray)
                    rgba_array = np.array(img_rgba)

                    # 创建保留掩膜
                    keep_mask = np.isin(mask_array, list(label_set))

                    # 根据背景模式处理像素
                    if background == "transparent":
                        # 非保留区域 alpha 设为 0（透明）
                        rgba_array[..., 3] = np.where(keep_mask, rgba_array[..., 3], 0)
                    elif background == "black":
                        # 非保留区域 RGB 设为 0（黑色），alpha 设为 255（不透明）
                        rgba_array[..., :3] = np.where(keep_mask[..., None], rgba_array[..., :3], 0)
                        rgba_array[..., 3] = 255
                    elif background == "white":
                        # 非保留区域 RGB 设为 255（白色），alpha 设为 255（不透明）
                        rgba_array[..., :3] = np.where(keep_mask[..., None], rgba_array[..., :3], 255)
                        rgba_array[..., 3] = 255

                    # 生成输出图像
                    out_img = Image.fromarray(rgba_array, 'RGBA')

                    # 准备保存参数，复制元数据
                    save_kwargs = {}
                    if 'exif' in img_info:
                        save_kwargs['exif'] = img_info['exif']
                    if 'icc_profile' in img_info:
                        save_kwargs['icc_profile'] = img_info['icc_profile']
                    if 'dpi' in img_info:
                        save_kwargs['dpi'] = img_info['dpi']
                    # 可以按需添加其他元数据
                    # if 'comment' in img_info:
                    #     save_kwargs['comment'] = img_info['comment']

                    # 保存输出图像
                    out_filename = base_name + output_ext
                    out_path = os.path.join(output_dir, out_filename)
                    out_img.save(out_path, **save_kwargs)

                    print(f"✅ 已处理：{image_filename} -> {out_filename}")
                    processed_count += 1

        except Exception as e:
            print(f"❌ 处理 {image_filename} 时出错：{e}")
            error_count += 1

    # 打印统计信息
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"✅ 成功处理：{processed_count} 个文件")
    print(f"⚠️  跳过：{skipped_count} 个文件")
    print(f"❌ 错误：{error_count} 个文件")
    print("=" * 50)

    return processed_count, skipped_count, error_count


# ==================== 执行处理 ====================
if __name__ == "__main__":
    # 执行提取
    extract_labels_from_mask(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        labels=LABELS,
        output_dir=OUTPUT_DIR,
        background=BACKGROUND,
        output_ext=OUTPUT_EXT,
        mask_ext=MASK_EXT
    )

    # 可选：显示处理结果的示例
    # 如果需要在 Notebook 中显示结果，取消下面的注释
    # from IPython.display import display
    # output_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*" + OUTPUT_EXT)))
    # if output_files:
    #     print("\n显示前 3 个处理结果：")
    #     for f in output_files[:3]:
    #         print(f"📷 {os.path.basename(f)}")
    #         display(Image.open(f))