#!/usr/bin/env python3
"""
批量修改文件扩展名（支持覆盖）
用法: python rename_ext.py <目录> <旧扩展名> <新扩展名> [--recursive]
示例: python rename_ext.py ./images jpg png
      python rename_ext.py ./images .jpg .png --recursive
"""

import os
import argparse
import sys

def rename_extensions(directory, old_ext, new_ext, recursive=False):
    """
    将 directory 下所有 old_ext 文件重命名为 new_ext，覆盖已有目标文件。
    """
    # 统一去除扩展名前的点号
    old_ext = old_ext.lstrip('.')
    new_ext = new_ext.lstrip('.')

    if not os.path.isdir(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        sys.exit(1)

    renamed_count = 0
    error_count = 0

    # 遍历文件
    for root, dirs, files in os.walk(directory):
        # 如果不是递归模式，只处理顶层目录，然后跳出循环
        if not recursive and root != os.path.abspath(directory):
            continue

        for filename in files:
            if filename.endswith(f'.{old_ext}'):
                old_path = os.path.join(root, filename)
                # 生成新文件名：去掉旧扩展名，添加新扩展名
                base = filename[:-len(old_ext)-1]  # 去除 .old_ext
                new_filename = f"{base}.{new_ext}"
                new_path = os.path.join(root, new_filename)

                try:
                    # os.replace 会原子性地替换文件，如果目标存在则直接覆盖
                    os.replace(old_path, new_path)
                    print(f"已重命名: {old_path} -> {new_path}")
                    renamed_count += 1
                except OSError as e:
                    print(f"重命名失败: {old_path} -> {new_path}，错误: {e}")
                    error_count += 1

    if renamed_count == 0 and error_count == 0:
        print(f"警告: 在 '{directory}' 中没有找到 .{old_ext} 文件")
    else:
        print(f"完成: 成功重命名 {renamed_count} 个文件", end='')
        if error_count > 0:
            print(f"，失败 {error_count} 个文件")
        else:
            print()

def main():
    parser = argparse.ArgumentParser(
        description="批量修改文件扩展名，覆盖目标文件"
    )
    parser.add_argument("--directory", help="目标目录路径",default=r"D:\DataBase\road\goose\val\labels\instanceids")
    parser.add_argument("--old_ext", help="原扩展名（如 jpg 或 .jpg）",default=".png")
    parser.add_argument("--new_ext", help="新扩展名（如 png 或 .png）",default=".jpg")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="递归处理子目录")
    args = parser.parse_args()

    rename_extensions(args.directory, args.old_ext, args.new_ext, args.recursive)

if __name__ == "__main__":
    main()