import os
import cv2
import numpy as np


def build_fixed_cmap():
    cmap = np.array([
        [0, 0, 0],          # 0 unlabeled
        [128, 64, 128],     # 1 paved-area
        [130, 76, 0],       # 2 dirt
        [0, 102, 0],        # 3 grass
        [112, 103, 87],     # 4 gravel
        [28, 42, 168],      # 5 water
        [48, 41, 30],       # 6 rocks
        [0, 50, 89],        # 7 pool
        [107, 142, 35],     # 8 vegetation
        [70, 70, 70],       # 9 roof
        [102, 102, 156],    # 10 wall
        [254, 228, 12],     # 11 window
        [254, 148, 12],     # 12 door
        [190, 153, 153],    # 13 fence
        [153, 153, 153],    # 14 fence-pole
        [255, 22, 96],      # 15 person
        [102, 51, 0],       # 16 dog
        [9, 143, 150],      # 17 car
        [119, 11, 32],      # 18 bicycle
        [51, 51, 0],        # 19 tree
        [190, 250, 190],    # 20 bald-tree
        [112, 150, 146],    # 21 ar-marker
        [2, 135, 115],      # 22 obstacle
        [255, 0, 0],        # 23 conflicting
    ], dtype=np.uint8)

    return cmap

def overlay_mask(image, mask, cmap, alpha=0.5):
    color_mask = cmap[mask]
    color_mask = color_mask[:, :, ::-1]  # RGB → BGR
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def build_image_index(img_dir):
    img_dict = {}
    for fname in os.listdir(img_dir):
        name, _ = os.path.splitext(fname)
        img_dict[name] = os.path.join(img_dir, fname)
    return img_dict


if __name__ == '__main__':

    mask_dir = r"D:\DataBase\archive\dataset\semantic_drone_dataset\label_images_semantic"
    img_dir  = r"D:\DataBase\archive\dataset\semantic_drone_dataset\original_images"

    cmap = build_fixed_cmap()

    # ===== 建立 image 索引 =====
    img_index = build_image_index(img_dir)

    for mask_name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_name)

        # ===== 读取 mask =====
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # ===== basename 匹配 =====
        base_name, _ = os.path.splitext(mask_name)

        if base_name not in img_index:
            print(f"[WARN] No image match for {mask_name}")
            continue

        image = cv2.imread(img_index[base_name])
        if image is None:
            print(f"[WARN] Failed to read image for {mask_name}")
            continue

        # ===== 尺寸对齐 =====
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # ===== overlay =====
        overlay = overlay_mask(image, mask, cmap, alpha=0.5)

        # ===== debug =====
        unique_values = np.unique(mask)
        print(f"{mask_name} classes:", unique_values)

        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask * 20)
        overlay=cv2.resize(overlay,(1080,640))
        cv2.imshow("overlay", overlay)

        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()