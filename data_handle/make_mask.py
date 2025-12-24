import os
import json
import cv2
import numpy as np

def generate_masks(
    images_dir,
    labels_dir,
    save_labels_dir,
    classes_map,
    generate_empty_mask=True,
    img_exts=('.jpg', '.jpeg', '.png', '.bmp')
):
    """
    根据 JSON 标注生成多分类 mask
    Args:
        images_dir: 原始图像文件夹
        labels_dir: JSON 标签文件夹
        save_labels_dir: mask 保存文件夹
        classes_map: dict, 类名 -> 类别值，例如 {'line':1,'bg':0}
        generate_empty_mask: 无 JSON 时是否生成全黑 mask
        img_exts: 支持的图像扩展名
    """
    os.makedirs(save_labels_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(img_exts)]

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ 图像读取失败：{img_path}")
            continue

        h, w = img.shape[:2]
        base_name = os.path.splitext(img_file)[0]
        json_path = os.path.join(labels_dir, base_name + '.json')

        # mask 初始化为 0（背景）
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

            print(f"✓ 生成 mask（含标注）：{img_file}")

        else:
            if generate_empty_mask:
                print(f"✗ 无 JSON → 生成全黑 mask：{img_file}")
            else:
                print(f"✗ 无 JSON → 跳过图像：{img_file}")
                continue

        # 保存 mask
        mask_save_path = os.path.join(save_labels_dir, base_name + ".png")
        cv2.imwrite(mask_save_path, mask)

    print("\n🎉 mask 生成完成！")


if __name__ == '__main__':
    # 类别映射示例：多分类 mask
    # 背景=0, 线=1, 圆=2, 方块=3 等
    classes_map = {
        'background': 0,
        'line': 255,
    }

    images_dir = r"D:\Projects\U-2-Net\make_train_datas\images"
    labels_dir = r"D:\Projects\U-2-Net\make_train_datas\labels"
    save_labels_dir = r"D:\Projects\U-2-Net\make_train_datas\masks"

    generate_masks(
        images_dir,
        labels_dir,
        save_labels_dir,
        classes_map,
        generate_empty_mask=True
    )
