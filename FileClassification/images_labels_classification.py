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

    参数:
        all_file_dir: str，源目录
        save_map: dict，类别名→路径与后缀映射
        num_workers: 并行线程数
        move: 是否移动文件（True 移动 / False 复制）
    """
    src_dir = Path(all_file_dir)
    if not src_dir.exists():
        print(f"❌ 源文件夹不存在: {src_dir}")
        return

    # 预建目录
    for v in save_map.values():
        Path(v['save_path']).mkdir(parents=True, exist_ok=True)

    # ===== 预计算后缀→路径映射表 =====
    suffix_to_path = {}
    for info in save_map.values():
        for suf in info['suffix']:
            suffix_to_path[suf.lower()] = info['save_path']

    # ===== 高效扫描文件（不递归太深） =====
    all_files = []
    for root, _, files in os.walk(src_dir):
        all_files += [Path(root) / f for f in files]
    total = len(all_files)
    print(f"📂 共发现文件 {total} 个，开始分类...")

    # ===== 定义任务函数 =====
    def process_file(file_path: Path):
        suf = file_path.suffix.lower()
        if suf not in suffix_to_path:
            return f"⚠️ 未匹配类型文件: {file_path}"
        dst_dir = Path(suffix_to_path[suf])
        dst_path = dst_dir / file_path.name
        try:
            if move:
                shutil.move(str(file_path), str(dst_path))
            else:
                shutil.copy2(str(file_path), str(dst_path))
            return None
        except Exception as e:
            return f"❌ 处理失败 {file_path}: {e}"

    # ===== 多线程执行 =====
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, f) for f in all_files]
        for i, fut in enumerate(as_completed(futures), 1):
            err = fut.result()
            if err:
                results.append(err)
            if i % 500 == 0:
                print(f"🧩 进度: {i}/{total}")

    if results:
        print("\n".join(results))
    print(f"✅ 分类完成，共处理 {total} 个文件。")



