import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
"""
将文件根据扩展名 分类复制或移动到指定目录
"""


def classification(all_file_dir, save_map, num_workers=8, move=False):
    """
    高速文件分类复制/移动
    """
    src_dir = Path(all_file_dir)
    if not src_dir.exists():
        print(f"源文件夹不存在: {src_dir}")
        return

    for v in save_map.values():
        Path(v['save_path']).mkdir(parents=True, exist_ok=True)

    suffix_to_path = {}
    for info in save_map.values():
        for suf in info['suffix']:
            suffix_to_path[suf.lower()] = info['save_path']

    all_files = []
    for root, _, files in os.walk(src_dir):
        all_files += [Path(root) / f for f in files]
    total = len(all_files)
    print(f"共发现文件 {total} 个，开始分类...")

    # 定义任务函数
    def process_file(file_path: Path):
        suf = file_path.suffix.lower()
        if suf not in suffix_to_path:
            return f"未匹配类型文件: {file_path}"
        dst_dir = Path(suffix_to_path[suf])
        dst_path = dst_dir / file_path.name
        try:
            if move:
                shutil.move(str(file_path), str(dst_path))
            else:
                shutil.copy2(str(file_path), str(dst_path))
            return None
        except Exception as e:
            return f"处理失败 {file_path}: {e}"

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, f) for f in all_files]
        for i, fut in enumerate(as_completed(futures), 1):
            err = fut.result()
            if err:
                results.append(err)
            if i % 500 == 0:
                print(f"进度: {i}/{total}")

    if results:
        print("\n".join(results))
    print(f"分类完成，共处理 {total} 个文件。")



if __name__ == '__main__':
    all_file_dir = r"D:\DataSets\Cable\datas"

    save_map = {
        'images': {'save_path':r'D:\DataSets\Cable\images', 'suffix': ['.jpg', '.png',".JPG"]},
        'json_labels': {'save_path': r"D:\DataSets\Cable\jsons", 'suffix': ['.json']},
        # 'txt_labels': {'save_path': r"D:\DataBase\cabel_train_datas\add_cable_datas\labels", 'suffix': ['.txt']}
    }

    #对文件进行分类
    classification(all_file_dir, save_map, num_workers=12, move=False)