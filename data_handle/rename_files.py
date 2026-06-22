#!/usr/bin/env python3
"""
批量重命名文件（修改扩展名 / 按时间戳或计数器重命名主文件名）
用法:
  仅改扩展名:
    python rename_files.py ./images jpg png
  按时间戳重命名:
    python rename_files.py ./images jpg png --mode timestamp
  按计数器重命名:
    python rename_files.py ./images jpg png --mode counter --base-name photo
  递归子目录:
    python rename_files.py ./images jpg png --recursive --mode counter
"""

import os
import sys
import argparse
from datetime import datetime

def natural_sort_key(s):
    """用于自然排序的辅助函数（使 '2' 排在 '10' 之前）"""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def rename_files(directory, old_ext, new_ext, recursive=False,
                 mode='none', base_name='new_name', time_format='%Y%m%d_%H%M%S',
                 start_num=1, num_digits=0):
    """
    批量重命名文件。
    - mode: 'none'  只修改扩展名
            'timestamp' 使用时间戳 + 序号作为新主文件名
            'counter'   使用 base_name + 序号作为新主文件名
    """
    old_ext = old_ext.lstrip('.')
    new_ext = new_ext.lstrip('.')

    if not os.path.isdir(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        sys.exit(1)

    # 收集所有需要重命名的文件路径
    target_files = []
    for root, dirs, files in os.walk(directory):
        if not recursive and root != os.path.abspath(directory):
            continue
        for f in files:
            if f.endswith(f'.{old_ext}'):
                target_files.append(os.path.join(root, f))

    if not target_files:
        print(f"警告: 在 '{directory}' 中没有找到 .{old_ext} 文件")
        return

    # 按路径（或文件名）排序，保证重命名顺序可预测
    target_files.sort(key=natural_sort_key)

    total = len(target_files)
    renamed_count = 0
    error_count = 0

    # 根据模式生成新文件名
    if mode == 'timestamp':
        # 同一批文件共用同一个时间戳，用序号保证唯一
        timestamp_str = datetime.now().strftime(time_format)
        # 自动计算序号宽度
        width = num_digits if num_digits > 0 else max(3, len(str(total)))
        for idx, old_path in enumerate(target_files, start=1):
            new_filename = f"{timestamp_str}_{idx:0{width}d}.{new_ext}"
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
            try:
                os.replace(old_path, new_path)
                print(f"已重命名: {old_path} -> {new_path}")
                renamed_count += 1
            except OSError as e:
                print(f"重命名失败: {old_path} -> {new_path}，错误: {e}")
                error_count += 1

    elif mode == 'counter':
        width = num_digits if num_digits > 0 else max(1, len(str(start_num + total - 1)))
        for idx, old_path in enumerate(target_files, start=start_num):
            new_filename = f"{base_name}{idx:0{width}d}.{new_ext}"
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
            try:
                os.replace(old_path, new_path)
                print(f"已重命名: {old_path} -> {new_path}")
                renamed_count += 1
            except OSError as e:
                print(f"重命名失败: {old_path} -> {new_path}，错误: {e}")
                error_count += 1

    else:  # mode == 'none'，只改扩展名
        for old_path in target_files:
            base = os.path.splitext(old_path)[0]  # 去掉旧扩展名的主路径
            new_path = base + '.' + new_ext
            try:
                os.replace(old_path, new_path)
                print(f"已重命名: {old_path} -> {new_path}")
                renamed_count += 1
            except OSError as e:
                print(f"重命名失败: {old_path} -> {new_path}，错误: {e}")
                error_count += 1

    # 汇总
    print(f"完成: 成功重命名 {renamed_count} 个文件", end='')
    if error_count:
        print(f"，失败 {error_count} 个文件")
    else:
        print()

def main():
    parser = argparse.ArgumentParser(
        description="批量修改文件扩展名（可同时重命名主文件名）")
    parser.add_argument("--directory", help="目标目录路径",default=r"E:\test\images")
    parser.add_argument("--old_ext", help="原扩展名（如 jpg 或 .jpg）",default=".jpg")
    parser.add_argument("--new_ext", help="新扩展名（如 png 或 .png）",default=".JPG")
    parser.add_argument( "--recursive", action="store_true",
                        help="递归处理子目录",default=True)
    parser.add_argument("--mode", choices=["none", "timestamp", "counter"],
                        default="none",
                        help="文件名重命名模式：none(仅改扩展名), timestamp(时间戳+序号), counter(基础名+序号)")
    parser.add_argument("--base-name", default="new_name",
                        help="counter 模式的基础名（默认: new_name）")
    parser.add_argument("--time-format", default="%Y%m%d_%H%M%S",
                        help="timestamp 模式的时间格式（默认: %%Y%%m%%d_%%H%%M%%S）")
    parser.add_argument("--start-num", type=int, default=1,
                        help="counter 模式的起始编号（默认: 1）")
    parser.add_argument("--num-digits", type=int, default=0,
                        help="编号位数，0 表示自动计算（默认: 0）")

    args = parser.parse_args()
    rename_files(args.directory, args.old_ext, args.new_ext, args.recursive,
                 args.mode, args.base_name, args.time_format,
                 args.start_num, args.num_digits)

if __name__ == "__main__":
    main()
