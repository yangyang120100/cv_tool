import os
from concurrent.futures import ThreadPoolExecutor

# 配置路径
labels_dir = r"D:\DataBase\person_datas\val\labels"          # 标签文件夹
out_dir = r"D:\DataBase\person_datas\val\re_labels"   # 输出修改后的标签
os.makedirs(out_dir, exist_ok=True)

# 标签映射规则，例如将1改为0
label_map = {1: 0}  # 可以扩展其他映射，如 {1:0, 2:5}

def modify_label_file(fname):
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
        # 修改标签
        if class_id in label_map:
            parts[0] = str(label_map[class_id])
        new_lines.append(" ".join(parts))

    # 写入文件
    with open(out_path, 'w') as f_out:
        f_out.write("\n".join(new_lines) + "\n")

# 多线程处理
max_workers = 16
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(modify_label_file, os.listdir(labels_dir))

print("标签修改完成！")
