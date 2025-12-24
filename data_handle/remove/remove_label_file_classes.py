import os
from concurrent.futures import ThreadPoolExecutor

# 配置路径
labels_dir = r"D:\DataBase\person_datas\val\labels"  # 标签文件夹
out_dir = r"D:\DataBase\person_datas\val\remove_labels"  # 输出过滤后的标签
os.makedirs(out_dir, exist_ok=True)

# 指定需要保留的类别 ID
keep_classes = {1}


def process_file(fname):
    if not fname.endswith('.txt'):
        return

    in_path = os.path.join(labels_dir, fname)
    out_path = os.path.join(out_dir, fname)

    with open(in_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        class_id = int(parts[0])
        if class_id in keep_classes:
            new_lines.append(line.strip())

    # 写入文件
    if new_lines:
        with open(out_path, 'w') as f_out:
            f_out.write("\n".join(new_lines) + "\n")
    else:
        open(out_path, 'w').close()


# 使用多线程
max_workers = 8  # 线程数，可根据 CPU 核心和磁盘情况调整
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_file, os.listdir(labels_dir))
