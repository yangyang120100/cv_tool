import os
from pathlib import Path

import cv2
import glob
import numpy as np
"""
将图像和 mask 保存为 npz 文件
"""

def npz_save(image_dir, label_dir, save_dir):
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    if len(image_files)>0 and len(label_files)>0:
        image_pather=Path(image_files[0])
        label_pather=Path(label_files[0])

        image_suffix=image_pather.suffix
        label_suffix=label_pather.suffix

        for image_file in image_files:
            image_path=os.path.join(image_dir,image_file)
            if not Path(image_file).stem+label_suffix in label_files:
                continue
            label_path = os.path.join(label_dir, Path(image_file).stem + label_suffix)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path)
            gray_label=cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
            binary_mask=np.where(gray_label>0,1,0).astype(np.uint8)

            np.savez_compressed(os.path.join(save_dir,Path(image_file).stem+'.npz'),image=image,mask=binary_mask)

if __name__ == '__main__':
    image_dir=r'D:\DataBase\line_train_dataset\augmented\train\images'
    label_dir=r'D:\DataBase\line_train_dataset\augmented\train\masks'
    save_dir=r'D:\DataBase\line_train_dataset\augmented\train\npz'

    npz_save(image_dir, label_dir, save_dir)
