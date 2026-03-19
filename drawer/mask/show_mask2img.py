import os
import cv2
import numpy as np

image_dir = r"D:\DataBase\cabel_train_datas\aug_images"
mask_dir = r"D:\DataBase\cabel_train_datas\aug_masks"
# save_dir = "overlay"
#
# os.makedirs(save_dir, exist_ok=True)

alpha = 0.5

# 生成随机颜色
def generate_colormap(num_classes=20):
    np.random.seed(42)
    colors = np.random.randint(0, 255, (num_classes, 3))
    colors[0] = [0, 0, 0]
    return colors

colors = generate_colormap(50)

# 支持的图片格式
exts = [".jpg", ".png", ".jpeg"]

# 建立mask索引
mask_dict = {}
for f in os.listdir(mask_dir):
    name, ext = os.path.splitext(f)
    if ext.lower() in exts:
        mask_dict[name] = os.path.join(mask_dir, f)

for f in os.listdir(image_dir):

    name, ext = os.path.splitext(f)

    if ext.lower() not in exts:
        continue

    if name not in mask_dict:
        print("mask not found:", name)
        continue

    img_path = os.path.join(image_dir, f)
    mask_path = mask_dict[name]

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    # 彩色mask
    color_mask = np.zeros_like(image)

    classes = np.unique(mask)

    for cls in classes:
        color_mask[mask == cls] = colors[cls]

    overlay = cv2.addWeighted(image, 1, color_mask, alpha, 0)

    cv2.imshow("overlay", cv2.resize(overlay,(1920,1080)))

    # 保存
    # save_path = os.path.join(save_dir, name + "_overlay.png")
    # cv2.imwrite(save_path, overlay)

    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()