import xml.etree.ElementTree as ET
import os
from tqdm import tqdm  # 如果没有请运行 pip install tqdm，或者删除相关代码


def convert_coordinates(size, box):
    """
    将 VOC 像素坐标转换为 YOLO 归一化比例坐标
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def batch_convert_voc_to_yolo(xml_dir, output_dir, class_list):
    """
    批量转换主函数
    """
    # 如果输出文件夹不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 建立类别映射表
    class_map = {name: i for i, name in enumerate(class_list)}

    # 获取目录下所有xml文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"检测到 {len(xml_files)} 个 XML 文件，开始转换...")

    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)
        txt_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        if size is None: continue
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 如果图片尺寸无效（如0），则跳过
        if w == 0 or h == 0: continue

        with open(txt_path, 'w', encoding='utf-8') as f:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in class_map:
                    continue

                # 可选：跳过标注为 difficult 的目标
                # if obj.find('difficult') is not None and obj.find('difficult').text == '1':
                #     continue

                cls_id = class_map[cls_name]
                xmlbox = obj.find('bndbox')

                # 提取坐标
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))

                # 转换坐标
                bb = convert_coordinates((w, h), b)

                # 写入文件：类别索引 x_center y_center width height
                f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

    # 生成 classes.txt 文件，方便后续训练参考
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        for cls in class_list:
            f.write(cls + '\n')

    print(f"转换成功！所有 TXT 文件已保存至: {output_dir}")


# ================= 配置区域 =================
if __name__ == "__main__":
    # 1. 你的 XML 文件夹路径
    input_xml_folder = r'E:\hat\xml'

    # 2. 转换后的 TXT 保存路径
    output_txt_folder = r'E:\hat\labels'

    # 3. 你的类别列表（顺序必须与你想要的索引一致）
    # 注意：即便你的 XML 里只有 hat，也建议写成列表形式
    my_classes = ["hat","person"]

    # 执行转换
    batch_convert_voc_to_yolo(input_xml_folder, output_txt_folder, my_classes)