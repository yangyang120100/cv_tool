import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def imread_unicode(file_path):
    with open(file_path, 'rb') as f:
        data=f.read()
    return cv2.imdecode(
        np.frombuffer(data,np.uint8),
        cv2.IMREAD_COLOR
    )

def imwrite_unicode(file_path, img):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    ext=os.path.splitext(file_path)[1]
    success,encoded_img=cv2.imencode(ext,img)
    if success:
        encoded_img.tofile(file_path)
        return True
    return False

def visualize_combined_obb(json_path_1, json_path_2, img_dir, output_dir, category_mapping, model_info_1, model_info_2):
    """
    同时读取两个COCO格式的OBB JSON文件，将对应的旋转框同时绘制在同一张原图上
    """
    os.makedirs(output_dir, exist_ok=True)

    print("正在扫描原图目录...")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"找不到原图路径: {img_dir}")
    disk_files = {f.lower(): f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}

    print("正在加载JSON文件...")
    with open(json_path_1, 'r', encoding='utf-8') as f:
        data_1 = json.load(f)
    with open(json_path_2, 'r', encoding='utf-8') as f:
        data_2 = json.load(f)

    def build_filename_to_anns(data):
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        filename_to_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id in id_to_filename:
                fname = id_to_filename[img_id]
                filename_to_anns.setdefault(fname, []).append(ann)
        return filename_to_anns

    anns_map_1 = build_filename_to_anns(data_1)
    anns_map_2 = build_filename_to_anns(data_2)

    all_filenames = set(anns_map_1.keys()).union(set(anns_map_2.keys()))

    print(f"开始可视化，共计 {len(all_filenames)} 张图片...")
    for file_name in tqdm(all_filenames):
        actual_file_name = file_name
        if file_name.lower() in disk_files:
            actual_file_name = disk_files[file_name.lower()]

        img_path = os.path.join(img_dir, actual_file_name)
        if not os.path.exists(img_path):
            print(f" 警告: 找不到图片 {img_path}，已跳过。")
            continue

        img = imread_unicode(img_path)
        if img is None:
            print(f" 警告: 无法读取图片 {img_path}，已跳过。")
            continue

        h, w, _ = img.shape
        font_scale = max(0.5, h / 1800.0)
        thickness = max(1, int(h / 900.0))
        box_thickness = max(2, int(h / 600.0))

        def draw_annotations(annotations, model_color):
            for ann in annotations:
                if 'segmentation' in ann and len(ann['segmentation']) > 0 and len(ann['segmentation'][0]) == 8:
                    pts = ann['segmentation'][0]
                elif 'bbox' in ann and len(ann['bbox']) == 8:
                    pts = ann['bbox']
                else:
                    continue

                pts = np.array(pts, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(img, [pts], isClosed=True, color=model_color, thickness=box_thickness)

                cat_id = ann['category_id']
                cat_name = category_mapping.get(cat_id, f"class_{cat_id}")

                if 'score' in ann:
                    cat_name = f"{cat_name} {ann['score']:.2f}"

                text_pos = (int(pts[0][0][0]), int(pts[0][0][1]) - 5)
                cv2.putText(img, cat_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale * 0.75, model_color, thickness, cv2.LINE_AA)

        if file_name in anns_map_1:
            draw_annotations(anns_map_1[file_name], model_info_1["color"])

        if file_name in anns_map_2:
            draw_annotations(anns_map_2[file_name], model_info_2["color"])

        start_x, start_y = int(20 * font_scale), int(20 * font_scale)
        padding = int(10 * font_scale)
        line_spacing = int(15 * font_scale)

        for model_info in [model_info_1, model_info_2]:
            text = model_info["name"]
            bg_color = model_info["color"]

            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            cv2.rectangle(img,
                          (start_x, start_y),
                          (start_x + text_w + padding * 2, start_y + text_h + padding * 2),
                          bg_color,
                          -1)

            cv2.putText(img,
                        text,
                        (start_x + padding, start_y + text_h + padding),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA)

            start_y += text_h + padding * 2 + line_spacing


        output_path = os.path.join(output_dir, actual_file_name)
        imwrite_unicode(output_path, img)


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 1. 核心路径配置
    JSON_PATH_1 = r"E:\opred_labels.json"  # 模型 1 (如 o-detr) 的 JSON 路径
    JSON_PATH_2 = r"E:\ypred_labels.json"  # 模型 2 (对比模型或GT) 的 JSON 路径
    IMAGE_DIR = r"D:\DataBase\Insulator_datas_add\images"  # 原始图像存放的文件夹路径
    OUTPUT_DIR = r"C:\Users\Kedio\Desktop\show"  # 绘制完成后结果的保存路径

    # 2. 手动定义类别映射关系 (根据你的数据集中 category_id 修改)
    CATEGORY_MAPPING = {
        1:"jyz_xsjyz",
        2:"jyz_zhusjyz",
        3:"jyz_pin",
        4:"jyz_porcelainceossarm",
        5:"jyz_shackle",
        6:"jyz_strain"
    }

    # 3. 模型标签与色彩配置 (注意：OpenCV中颜色通道顺序为 BGR)
    MODEL_INFO_1 = {
        "name": "o-detr",
        "color": (255, 0, 0)  # 纯蓝色 (B=255, G=0, R=0)
    }
    MODEL_INFO_2 = {
        "name": "yolo",  # 另一个模型或标签的显示名称
        "color": (0, 0, 255)  # 纯红色 (B=0, G=0, R=255)
    }
    # ==================================================

    # 执行可视化任务
    visualize_combined_obb(
        json_path_1=JSON_PATH_1,
        json_path_2=JSON_PATH_2,
        img_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        category_mapping=CATEGORY_MAPPING,
        model_info_1=MODEL_INFO_1,
        model_info_2=MODEL_INFO_2
    )
    print(" 可视化合并图片全部生成完毕！")