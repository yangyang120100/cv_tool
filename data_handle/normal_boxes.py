import os
from PIL import Image
"""
将 LabelMe 格式的标注转换为 YOLO 格式
"""
def convert_label_line(line, img_width, img_height):
    parts = line.strip().split()
    if len(parts) != 5 and len(parts) != 6:
        raise ValueError(f"Invalid label line: {line}")
    # 假设格式: class_id x_min y_min x_max y_max
    if len(parts) == 6:
        class_id = parts[0]
        x_min = float(parts[2])
        y_min = float(parts[3])
        x_max = float(parts[4])
        y_max = float(parts[5])
    else:
        class_id = parts[0]
        x_min = float(parts[1])
        y_min = float(parts[2])
        x_max = float(parts[3])
        y_max = float(parts[4])

    # 转 YOLO 相对坐标
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    box_width = x_max - x_min
    box_height = y_max - y_min

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = box_width / img_width
    h_norm = box_height / img_height

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

def convert_file(txt_path, img_path, out_txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    img = Image.open(img_path)
    w, h = img.size

    with open(out_txt_path, 'w') as f_out:
        for line in lines:
            if line.strip() == "":
                continue
            try:
                new_line = convert_label_line(line, w, h)
                f_out.write(new_line + "\n")
            except Exception as e:
                print(f"Skipping line due to error: {line.strip()} -> {e}")

if __name__ == "__main__":
    label_dir = r"D:\DataBase\WiderPerson\labels"  # 标签文件夹
    img_dir = r"D:\DataBase\WiderPerson\Images"    # 图像文件夹
    out_dir = r"D:\DataBase\WiderPerson\yolo_labels"  # 输出 YOLO 标签文件夹
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            base = fname[:-4]
            txt_path = os.path.join(label_dir, fname)
            # 在图像文件夹查找同名图像
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_path = os.path.join(img_dir, base + ext)
                if os.path.exists(img_path):
                    out_txt_path = os.path.join(out_dir, fname)
                    print(f"Converting {txt_path} + {img_path} -> {out_txt_path}")
                    convert_file(txt_path, img_path, out_txt_path)
                    break
            else:
                print(f"WARNING: image file for {fname} not found, skip.")
