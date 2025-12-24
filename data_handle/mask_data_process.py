import os

import cv2
import numpy as np



# def mask_data_process(mask_path, save_path):
#     for file in os.listdir(mask_path):
#         img_dir = os.path.join(mask_path, file)
#         img = cv2.imread(img_dir)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # 将大于0的灰度值转为1
#         binary_mask = np.where(gray > 0, 1, 0).astype(np.uint8)
#         cv2.imwrite(os.path.join(save_path, file), binary_mask)


if __name__ == '__main__':
    mask_data_dir=r"D:\DataBase\test_Swin-unet_train_datas\gts-20251111T092834Z-1-001"
    save_mask_dir=r"D:\DataBase\test_Swin-unet_train_datas\process_mask"


    # mask_data_process(mask_data_dir,save_mask_dir)


