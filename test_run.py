# path = r"C:\Users\15721\Desktop\new_classes_map.txt"
#
# classes_map = dict()
# with open(path, encoding='utf-8') as w:
#     t_datas = w.readlines()
#     count = 0
#     for t in t_datas:
#         t = t.strip()          # 去除换行符及两端空白
#         # 或者 t = t.rstrip('\n')   # 仅去除末尾换行符
#         if t:                  # 可选：跳过空行
#             classes_map[t] = count
#             count += 1
# print(classes_map)
# print(classes_map.keys())

import os
import cv2
from pathlib import Path

image_dir=r"E:\MyRoad\images"
mask_dir=r"E:\MyRoad\masks"

for image_name in os.listdir(image_dir):
    image_path=os.path.join(image_dir,image_name)
    mask_path=os.path.join(mask_dir,Path(image_name).stem+".png")
    if not os.path.exists(mask_path):
        print(image_name)
        print(mask_path)
        print('\n')
    # image_shape=cv2.imread(image_path).shape[0:2]
    # mask_shape=cv2.imread(mask_path).shape[0:2]
    # if image_shape!=mask_shape:
    #     print(image_name)

