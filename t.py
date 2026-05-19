#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本：get_dir_sizes.py
用法：python get_dir_sizes.py /path/to/directory
功能：列出指定目录下所有子文件夹及其占用的磁盘空间大小（递归统计）。
"""

import os
import sys
import argparse
from pathlib import Path

def human_readable_size(size_bytes: int) -> str:
    """将字节数转换为人类可读的字符串（如 '1.23 MB'）。"""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while size_bytes >= 1024 and idx < len(units) - 1:
        size_bytes /= 1024.0
        idx += 1
    # 根据数值大小决定小数位数
    if idx == 0:
        return f"{int(size_bytes)} {units[idx]}"
    else:
        return f"{size_bytes:.2f} {units[idx]}"

def get_folder_size(path: Path) -> int:
    """
    递归计算文件夹的总大小（字节数）。
    忽略无法访问的文件或文件夹（如权限不足）。
    """
    total = 0
    try:
        # 使用 scandir 比 walk 更高效
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    try:
                        total += entry.stat().st_size
                    except (OSError, PermissionError):
                        continue  # 跳过无法读取大小的文件
                elif entry.is_dir(follow_symlinks=False):
                    # 递归计算子文件夹大小
                    total += get_folder_size(Path(entry.path))
    except PermissionError:
        # 无法读取该文件夹，返回 0
        pass
    return total

def main():
    parser = argparse.ArgumentParser(
        description="列出指定目录下所有子文件夹及其占用的磁盘空间"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=r"D:\Projects",
        help="要扫描的目录路径"
    )
    parser.add_argument(
        "-s", "--sort",
        action="store_true",
        help="按大小降序排序（默认按文件夹名称排序）"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="同时显示隐藏文件夹（Unix 下以 '.' 开头，Windows 下隐藏属性）"
    )
    args = parser.parse_args()

    root_dir = Path(args.directory).resolve()
    if not root_dir.is_dir():
        print(f"错误：'{root_dir}' 不是有效的目录或不存在。")
        sys.exit(1)

    print(f"正在扫描目录: {root_dir}")
    print("-" * 60)

    # 收集所有直接子文件夹
    # 方法：使用 iterdir() 并检查 is_dir()
    # 如果不需要隐藏文件夹，过滤掉名称以 '.' 开头的（Unix 风格）
    folders = []
    for item in root_dir.iterdir():
        if item.is_dir():
            if not args.all and item.name.startswith('.'):
                continue
            folders.append(item)

    if not folders:
        print("该目录下没有找到任何子文件夹。")
        return

    # 计算每个文件夹的大小（可能需要一些时间）
    results = []
    for folder in folders:
        print(f"计算中: {folder.name} ...", end=' ', flush=True)
        size_bytes = get_folder_size(folder)
        size_human = human_readable_size(size_bytes)
        results.append((folder.name, size_bytes, size_human))
        print("完成")

    # 排序
    if args.sort:
        results.sort(key=lambda x: x[1], reverse=True)  # 按字节数降序
    else:
        results.sort(key=lambda x: x[0])  # 按文件夹名升序

    # 输出表格
    print("\n" + "-" * 60)
    # 计算最大名称长度用于对齐
    max_name_len = max(len(name) for name, _, _ in results)
    name_width = min(max_name_len, 50)  # 限制最大宽度
    for name, _, size_human in results:
        # 对名称进行截断（如果需要）
        display_name = name if len(name) <= name_width else name[:name_width-3] + "..."
        print(f"{display_name:<{name_width}}  {size_human:>12}")
    print("-" * 60)

if __name__ == "__main__":
    main()