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

        # 类别颜色表（BGR）
        color_map = {
            0: (0, 0, 0),  # background
            1: (0, 255, 0),  # cable - green
            2: (255, 0, 0),  # class2 - blue
            3: (0, 0, 255),  # class3 - red
        }

        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id, color in color_map.items():
            color_mask[mask == cls_id] = color

        cv2.imshow("Color Mask (Class-wise)", color_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    masks_dir=r"D:\DataBase\Transmission_Tower\train\masks"
    show_pseudo_color(masks_dir)