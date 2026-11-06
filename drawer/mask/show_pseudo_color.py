import os

import cv2
import numpy as np


def show_pseudo_color(masks_dir):
    for file_name in os.listdir(masks_dir):
        mask_file_path=os.path.join(masks_dir, file_name)
        mask = cv2.imread(
            mask_file_path,
            cv2.IMREAD_GRAYSCALE
        )

        color_map = {
            0: (0, 0, 0),
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (0, 0, 255),
            4: (255,0,255),
        }

        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id, color in color_map.items():
            color_mask[mask == cls_id] = color                                

        cv2.imshow("Color Mask (Class-wise)", cv2.resize(color_mask,(1920,1080)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    masks_dir=r"D:\DataBase\cabel_train_datas\masks"
    show_pseudo_color(masks_dir)