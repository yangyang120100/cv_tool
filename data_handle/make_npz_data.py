import os
import argparse
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

    gray_label = cv_imread_cn(label_path, cv2.IMREAD_GRAYSCALE)
    if gray_label is None:
        print(f"⚠ mask 读取失败：{label_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, Path(image_file).stem + '.npz')

    np.savez_compressed(save_path, image=image, mask=gray_label)
    print(f"✓ {image_file} 保存完成")
    # print(f"image_file:{image_file}save done")


def npz_save(image_dir, label_dir, save_dir,max_workers):
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    if len(image_files) > 0 and len(label_files) > 0:
        image_pather = Path(image_files[0])
        label_pather = Path(label_files[0])

        image_suffix = image_pather.suffix
        label_suffix = label_pather.suffix

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_file in image_files:
                executor.submit(process_single_file, image_file, image_dir, label_dir, save_dir, label_suffix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default=r'D:\DataBase\Insulator_datas\images', help='图像路径')
    parser.add_argument('--label_dir', type=str,
                        default=r'D:\DataBase\Insulator_datas\masks', help='掩码图路径')
    parser.add_argument('--save_dir', type=str,
                        default=r'D:\DataBase\Insulator_datas\npz', help='保存路径')
    parser.add_argument('--max_workers', type=str,
                        default=32, help='保存线程数')
    args = parser.parse_args()

    npz_save(args.image_dir, args.label_dir, args.save_dir,args.max_workers)

