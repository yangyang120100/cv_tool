import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# 固定类别映射
# =============================
CLASS_MAP = {
    "jyz_xsjyz": 0,
    "jyz_zhusjyz": 1,
    "jyz_pin": 2,
    "jyz_porcelaincrossarm": 3,
    "jyz_shackle": 4,
    "jyz_strain": 5
}


def convert_obb_to_hbb(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return (
        x_center / img_w,
        y_center / img_h,
        width / img_w,
        height / img_h
    )


def process_json(json_path, save_dir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]
        shapes = data["shapes"]

        txt_name = os.path.splitext(os.path.basename(json_path))[0] + ".txt"
        save_path = os.path.join(save_dir, txt_name)

        with open(save_path, 'w', encoding='utf-8') as f:
            for shape in shapes:
                label = shape["label"]

                if label not in CLASS_MAP:
                    continue

                class_id = CLASS_MAP[label]
                points = shape["points"]

                xmin, ymin, xmax, ymax = convert_obb_to_hbb(points)
                x_center, y_center, w, h = convert_to_yolo(
                    xmin, ymin, xmax, ymax, img_w, img_h
                )

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        return True, json_path

    except Exception as e:
        return False, f"{json_path} -> {e}"


def main(json_dir, save_dir, num_workers=8):
    os.makedirs(save_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    total = len(json_files)

    print(f"共发现 {total} 个 JSON 文件")
    print(f"使用线程数: {num_workers}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_json, j, save_dir) for j in json_files]

        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()

            if success:
                print(f"[{i}/{total}] ✅ {os.path.basename(msg)}")
            else:
                print(f"[{i}/{total}] ❌ {msg}")

    # 写 classes.txt
    class_file = os.path.join(save_dir, "classes.txt")
    with open(class_file, 'w', encoding='utf-8') as f:
        for cls_name in CLASS_MAP:
            f.write(cls_name + "\n")

    print("🎉 全部转换完成！")


if __name__ == "__main__":
    json_dir = r"D:\DataBase\Insulator_datas\jsons"     # OBB json目录
    save_dir = r"D:\DataBase\Insulator_datas\labels"    # 输出YOLO标签目录
    main(json_dir, save_dir, num_workers=8)  # 可改线程数