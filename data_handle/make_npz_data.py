import os
from pathlib import Path

import cv2
import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""
将图像和 mask 保存为 npz 文件
"""

def process_single_file(image_file, image_dir, label_dir, save_dir, label_suffix):
    image_path = os.path.join(image_dir, image_file)
    label_file = Path(image_file).stem + label_suffix
    if not label_file in os.listdir(label_dir):
        return
    label_path = os.path.join(label_dir, label_file)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(label_path)
    gray_label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    binary_mask = np.where(gray_label > 0, 1, 0).astype(np.uint8)

    np.savez_compressed(os.path.join(save_dir, Path(image_file).stem + '.npz'), image=image, mask=binary_mask)

def npz_save(image_dir, label_dir, save_dir):
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    if len(image_files) > 0 and len(label_files) > 0:
        image_pather = Path(image_files[0])
        label_pather = Path(label_files[0])

        image_suffix = image_pather.suffix
        label_suffix = label_pather.suffix

        with ThreadPoolExecutor(max_workers=4) as executor:
            for image_file in image_files:
                executor.submit(process_single_file, image_file, image_dir, label_dir, save_dir, label_suffix)

if __name__ == '__main__':
    image_dir = r'D:\DataBase\cabel_train_datas\augmented_wash\val\images'
    label_dir = r'D:\DataBase\cabel_train_datas\augmented_wash\val\masks'
    save_dir = r'D:\DataBase\cabel_train_datas\augmented_wash\val\npz'

    npz_save(image_dir, label_dir, save_dir)
