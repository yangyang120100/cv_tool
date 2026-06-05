import os
import cv2
import numpy as np

#修改语义分割的像素值（修改分类）

THING_CLASSES_ID = [1, 6, 10, 12, 13, 14, 15, 19, 20, 25, 28, 32, 33, 34, 35, 36, 37, 40, 45, 46, 47, 49, 57, 58, 60, 63]
STUFF_CLASSES_ID = [0, 2, 3, 4, 5, 7, 8, 9, 11, 16, 17, 18, 21, 22, 23, 24, 26, 27, 29, 30, 31, 38, 39, 41, 42, 43, 44, 48, 50, 51, 52, 53, 54, 55, 56, 59, 61, 62]

stuff_map = {old_id: new_id for new_id, old_id in enumerate(STUFF_CLASSES_ID)}
thing_set = set(THING_CLASSES_ID)

src_root = r"D:\DataBase\road\goose\val\labels\semantic"
dst_root = r"D:\DataBase\road\goose\val\labels\semantic_re"
ignore_label = 255

os.makedirs(dst_root, exist_ok=True)

for fn in os.listdir(src_root):
    if not fn.lower().endswith(".png"):
        continue

    src_path = os.path.join(src_root, fn)
    dst_path = os.path.join(dst_root, fn)

    mask = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("skip unreadable:", fn)
        continue

    new_mask = np.full(mask.shape, ignore_label, dtype=np.uint8)

    for old_id, new_id in stuff_map.items():
        new_mask[mask == old_id] = new_id

    # thing 全部保留为 255
    for old_id in thing_set:
        new_mask[mask == old_id] = ignore_label

    cv2.imwrite(dst_path, new_mask)

print("done")