import os
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def cv_imread_cn(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv_imwrite_cn(path, img):
    ext = os.path.splitext(path)[1]
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(path)
    return success

def process_single_image(
    img_file,
    images_dir,
    labels_dir,
    save_labels_dir,
    classes_map,
    generate_empty_mask
):
    img_path = os.path.join(images_dir, img_file)
    img = cv_imread_cn(img_path)
    if img is None:
        return f"⚠ 图像读取失败：{img_path}"

    h, w = img.shape[:2]
    base_name = os.path.splitext(img_file)[0]
    json_path = os.path.join(labels_dir, base_name + '.json')

    mask = np.zeros((h, w), dtype=np.uint8)

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        shapes = json_data.get('shapes', [])
        for shape in shapes:
            label_name = shape['label']
            if label_name in classes_map:
                points = np.array(shape['points'], dtype=np.int32)
                class_value = classes_map[label_name]
                cv2.fillPoly(mask, [points], class_value)

        status = f"✓ 生成 mask（含标注）：{img_file}"

    else:
        if not generate_empty_mask:
            return f"✗ 无 JSON → 跳过：{img_file}"
        status = f"✗ 无 JSON → 生成全黑 mask：{img_file}"

    mask_save_path = os.path.join(save_labels_dir, base_name + ".png")
    cv_imwrite_cn(mask_save_path, mask)

    return status


def generate_masks_multithread(
    images_dir,
    labels_dir,
    save_labels_dir,
    classes_map,
    generate_empty_mask=True,
    img_exts=('.jpg', '.jpeg', '.png', '.bmp'),
    num_workers=8
):
    os.makedirs(save_labels_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(img_exts)
    ]

    print(f"启动多线程：{num_workers} workers，共 {len(image_files)} 张图像\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                img_file,
                images_dir,
                labels_dir,
                save_labels_dir,
                classes_map,
                generate_empty_mask
            )
            for img_file in image_files
        ]

        for future in as_completed(futures):
            print(future.result())

    print("\nmask 生成完成！")

if __name__ == '__main__':
    # classes_map = {
    #     'background': 0,
    #     'cable':1,
    #     'white_cable':2
    #     # 'pole': 1,
    #     # 'pole_steelpipe': 2,
    #     # 'pole_jiaogangta': 3
    # }
    # classes_map = {
    # 'jyz_xsjyz':0,
    # 'jyz_zhusjyz':1,
    # 'jyz_pin':2,
    # 'jyz_porcelaincrossarm':3,
    # 'jyz_shackle':4,
    # 'jyz_strain':5
    # }
    classes_map={'background': 0, 'asphalt-road': 1, 'concrete-pavement': 2, 'brick-road': 3, 'dirt': 4, 'gravel': 5, 'inland-water': 6, 'standing-water': 7, 'woods': 8, 'tall-vegetation': 9, 'low-vegetation': 10, 'roof': 11, 'wall': 12, 'fence': 13, 'fence-post': 14, 'snow': 15, 'concrete-pole': 16, 'angle-steel-tower': 17, 'steel-tube-tower': 18, 'conductor': 19, 'sky': 20, 'obstacle': 21, 'car': 22, 'bus': 23, 'work-vehicle': 24, 'large-vehicle': 25, 'bicycle': 26, 'person': 27, 'manhole-cover': 28, 'distant-building': 29, 'traffic-light': 30, 'house': 31, 'tricycle': 32}

    images_dir = r"D:\DataSets\MyRoad\images"
    labels_dir = r"D:\DataSets\MyRoad\jsons"
    save_labels_dir = r"D:\DataSets\MyRoad\masks"

    generate_masks_multithread(
        images_dir,
        labels_dir,
        save_labels_dir,
        classes_map,
        generate_empty_mask=True,
        num_workers=16 #线程数
    )