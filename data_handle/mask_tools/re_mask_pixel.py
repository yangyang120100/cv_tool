import os
import numpy as np
from PIL import Image


def remap_mask(mask: np.ndarray, mapping: dict) -> np.ndarray:
    """
    根据字典映射修改 mask 中的像素值。

    参数:
        mask: 输入的分割 mask，形状为 (H, W) 的 numpy 数组，整数类型。
        mapping: 像素值映射字典，格式为 {原像素值: 新像素值}。

    返回:
        重新映射后的 numpy 数组，形状与输入相同。
    """
    remapped = mask.copy()
    for old_val, new_val in mapping.items():
        remapped[mask == old_val] = new_val
    return remapped


def remap_mask_file(
    input_path: str,
    output_path: str,
    mapping: dict
) -> None:
    """
    读取单个图像文件，修改像素值后保存。

    参数:
        input_path: 输入 mask 图像路径。
        output_path: 输出图像保存路径。
        mapping: 像素值映射字典。
    """
    mask = np.array(Image.open(input_path))
    new_mask = remap_mask(mask, mapping)
    # 保存为灰度图，若需保留调色板可进一步处理
    Image.fromarray(new_mask.astype(np.uint8)).save(output_path)


def remap_folder(
    input_dir: str,
    output_dir: str,
    mapping: dict,
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
) -> None:
    """
    批量处理文件夹中所有 mask 图像，将映射后的结果保存到输出文件夹。

    参数:
        input_dir: 存放原始 mask 的文件夹路径。
        output_dir: 保存处理后 mask 的文件夹路径（自动创建）。
        mapping: 像素值映射字典。
        extensions: 需要处理的图像文件后缀名元组。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f'处理: {filename}')
            try:
                remap_mask_file(input_path, output_path, mapping)
            except Exception as e:
                print(f'处理 {filename} 时出错: {e}')


if __name__ == "__main__":
    # 示例：将值为 1 的类别改为 255，值为 2 的改为 128
    mapping_dict = {0:255}

    # 批量处理文件夹
    input_folder = r"D:\DataSets\MyRoad\copy_remasks"   # 替换为你的输入文件夹路径
    output_folder = r"D:\DataSets\MyRoad\copy_remasks" # 替换为你的输出文件夹路径

    remap_folder(input_folder, output_folder, mapping_dict)
    print("批量处理完成！")