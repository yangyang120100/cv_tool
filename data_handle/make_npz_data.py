import os
from pathlib import Path

import cv2
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""
将图像和 mask 保存为 npz 文件
"""

def cv_imread_cn(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def cv_imwrite_cn(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(path)
    return success

def process_single_file(image_file, image_dir, label_dir, save_dir, label_suffix):
    image_path = os.path.join(image_dir, image_file)
    label_file = Path(image_file).stem + label_suffix

    if not os.path.exists(os.path.join(label_dir, label_file)):
        return

    label_path = os.path.join(label_dir, label_file)

    image = cv_imread_cn(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"⚠ 读取失败：{image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray_label = cv_imread_cn(label_path, cv2.IMREAD_GRAYSCALE)
    if gray_label is None:
        print(f"⚠ mask 读取失败：{label_path}")
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, Path(image_file).stem + '.npz')

    np.savez_compressed(save_path, image=image, mask=gray_label)
    print(f"✓ {image_file} 保存完成")
    # print(f"image_file:{image_file}save done")


def npz_save(image_dir, label_dir, save_dir):
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    if len(image_files) > 0 and len(label_files) > 0:
        image_pather = Path(image_files[0])
        label_pather = Path(label_files[0])

        image_suffix = image_pather.suffix
        label_suffix = label_pather.suffix

        with ThreadPoolExecutor(max_workers=16) as executor:
            for image_file in image_files:
                executor.submit(process_single_file, image_file, image_dir, label_dir, save_dir, label_suffix)

if __name__ == '__main__':
    image_dir = r'D:\DataBase\Transmission_Tower\train\images'
    label_dir = r'D:\DataBase\Transmission_Tower\train\masks'
    save_dir = r'D:\DataBase\Transmission_Tower\train\npz'

    npz_save(image_dir, label_dir, save_dir)