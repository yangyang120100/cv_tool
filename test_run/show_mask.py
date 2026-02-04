import os
import cv2
import numpy as np


def voc_colormap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = np.array([r, g, b])
    return cmap

def rgb_to_mask(rgb_label, cmap):
    h, w, _ = rgb_label.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for class_id, color in enumerate(cmap):
        matches = np.all(rgb_label == color, axis=-1)
        mask[matches] = class_id

    return mask

if __name__ == '__main__':
    mask_dir = r"D:\DataBase\Transmission_Tower\train\masks"

    for mask_file_name in os.listdir(mask_dir):
        mask_file_path = os.path.join(mask_dir, mask_file_name)
        mask=cv2.imread(mask_file_path,cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('mask',mask*100)
        # cv2.waitKey(0)

        # mask = cv2.imread(mask_file_path)[:, :, ::-1]
        # cmap = voc_colormap()
        # mask = rgb_to_mask(mask, cmap)
        # # # 输出mask中的像素种类
        unique_values = np.unique(mask)
        # print(f"{mask_file_name} unique pixel values:", unique_values)
        print(unique_values)

        #
        # cv2.imshow('mask', mask * 255 if len(unique_values) == 2 else mask)
        # cv2.waitKey(0)
