import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm


def voc_to_coco(xml_path, json_save_path, class_names):
    """
    Args:
        xml_path: 存放 XML 文件的文件夹路径
        json_save_path: 输出 JSON 文件的路径
        class_names: 类别列表，例如 ['dog', 'cat', 'person']
    """
    dataset = {
        "info": {"description": "VOC to COCO Dataset", "year": 2024},
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充 categories
    for i, name in enumerate(class_names):
        dataset["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "none"
        })

    ann_id = 1
    img_id = 1

    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc="Converting"):
        tree = ET.parse(os.path.join(xml_path, xml_file))
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        filename = root.find('filename').text

        # 添加图像信息
        dataset["images"].append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": img_id
        })

        # 遍历所有目标框
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_names:
                continue

            cls_id = class_names.index(name)
            xmlbox = obj.find('bndbox')

            # VOC: [xmin, ymin, xmax, ymax]
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)

            # COCO: [x_min, y_min, width, height]
            w = xmax - xmin
            h = ymax - ymin

            dataset["annotations"].append({
                "segmentation": [],  # 水平框通常不需要分割数据
                "area": w * h,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [xmin, ymin, w, h],
                "category_id": cls_id,
                "id": ann_id
            })
            ann_id += 1

        img_id += 1

    # 保存文件
    with open(json_save_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"\nSuccessfully saved to {json_save_path}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 填入你的类别名称（顺序需固定）
    MY_CLASSES = ['hat', 'person']

    # 2. 设置路径
    XML_DIR = r'E:\hat\xml'  # XML文件夹路径
    SAVE_JSON = r'E:\hat\labels\save.json'  # 保存的JSON路径

    voc_to_coco(XML_DIR, SAVE_JSON, MY_CLASSES)