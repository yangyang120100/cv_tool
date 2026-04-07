







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量重命名包含中文字符的文件对（图像+标签），保持配对文件的基本名一致。
用法：
    python rename_chinese_pairs.py /path/to/images /path/to/labels
可选参数：
    --new-name-prefix 指定新文件名前缀（默认将删除中文字符，保留字母数字等）
    --dry-run 只显示将要执行的操作，不实际重命名
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

def contains_chinese(text: str) -> bool:
    """检测字符串是否包含中文字符（CJK统一表意文字）"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def generate_new_name(old_stem: str, prefix: str = None, counter: int = None) -> str:
    """
    根据旧文件名生成新文件名（不含扩展名）。
    - 若指定 prefix，则新名称为 prefix + (counter if counter else '')
    - 否则，移除所有中文字符，只保留字母、数字、点、下划线、连字符，
      如果结果为空则使用 "file"
    """
    if prefix is not None:
        if counter is not None:
            return f"{prefix}{counter}"
        else:
            return prefix
    else:
        # 移除中文字符，保留常用安全字符
        cleaned = re.sub(r'[\u4e00-\u9fff]', '', old_stem)
        cleaned = re.sub(r'[^\w\-\.]', '', cleaned)  # 移除其他特殊字符
        if not cleaned:
            cleaned = "file"
        return cleaned

def find_files_with_chinese(folder: Path) -> dict:
    """返回字典：{基本名: 文件路径}，仅包含文件名含中文的文件"""
    result = {}
    if not folder.is_dir():
        print(f"错误：{folder} 不是有效目录")
        return result
    for file in folder.iterdir():
        if file.is_file():
            stem = file.stem
            if contains_chinese(stem):
                # 处理同名文件覆盖问题（理论上同一文件夹内不应有相同基本名，但以防万一）
                if stem in result:
                    print(f"警告：文件夹 {folder} 中发现多个同名文件（基本名相同），将只保留第一个：{file}")
                else:
                    result[stem] = file
    return result

def main():
    parser = argparse.ArgumentParser(description="批量重命名中文字符文件对（图像+标签）")
    parser.add_argument("--image_dir", default=r"D:\DataBase\Insulator_datas\images", help="图像文件夹路径")
    parser.add_argument("--label_dir", default=r"D:\DataBase\Insulator_datas\jsons", help="标签文件夹路径")
    parser.add_argument("--new-name-prefix", default=None,
                        help="指定新文件名前缀（不含扩展名），不指定则自动移除中文字符")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要执行的操作，不实际重命名")
    args = parser.parse_args()

    image_path = Path(args.image_dir)
    label_path = Path(args.label_dir)

    # 1. 分别找出两个文件夹中包含中文命名的文件
    images = find_files_with_chinese(image_path)
    labels = find_files_with_chinese(label_path)

    if not images and not labels:
        print("两个文件夹中均未找到包含中文命名的文件。")
        return

    # 2. 匹配相同基本名的文件对
    common_stems = set(images.keys()) & set(labels.keys())
    if not common_stems:
        print("未找到匹配的文件对（基本名相同且包含中文）。")
        return

    # 3. 准备重命名映射
    # 为避免重名冲突，需要为每个文件对生成唯一的新名称
    rename_actions = []  # (old_path, new_path)

    # 若用户提供了前缀，则生成唯一名称；否则按原规则（可能多个文件对产生相同的新名称）
    if args.new_name_prefix:
        # 使用前缀+序号，保证唯一
        for idx, stem in enumerate(sorted(common_stems), start=1):
            new_stem = generate_new_name(stem, prefix=args.new_name_prefix, counter=idx)
            img_file = images[stem]
            lbl_file = labels[stem]
            new_img = img_file.with_stem(new_stem)
            new_lbl = lbl_file.with_stem(new_stem)
            rename_actions.append((img_file, new_img))
            rename_actions.append((lbl_file, new_lbl))
    else:
        # 自动清理中文字符，但可能产生重复，需要额外处理
        # 收集所有待生成的新基本名，若重复则追加序号
        stem_counter = defaultdict(int)
        for stem in common_stems:
            new_stem_base = generate_new_name(stem)
            stem_counter[new_stem_base] += 1
        # 先确定每个 stem 的最终新名（带序号避免冲突）
        final_new_names = {}
        for stem in common_stems:
            base = generate_new_name(stem)
            if stem_counter[base] > 1:
                # 需要加序号
                # 按原始 stem 排序保证确定性
                idx = 1
                for s in sorted(common_stems):
                    if s == stem:
                        break
                    if generate_new_name(s) == base:
                        idx += 1
                new_stem = f"{base}{idx}"
            else:
                new_stem = base
            final_new_names[stem] = new_stem

        for stem in common_stems:
            new_stem = final_new_names[stem]
            img_file = images[stem]
            lbl_file = labels[stem]
            new_img = img_file.with_stem(new_stem)
            new_lbl = lbl_file.with_stem(new_stem)
            rename_actions.append((img_file, new_img))
            rename_actions.append((lbl_file, new_lbl))

    # 检查目标文件是否已存在（避免覆盖）
    conflicts = []
    for old, new in rename_actions:
        if new.exists() and new != old:
            conflicts.append((old, new))
    if conflicts:
        print("以下目标文件已存在，无法重命名（请手动处理或使用 --new-name-prefix 指定唯一前缀）：")
        for old, new in conflicts:
            print(f"  {old} -> {new}")
        sys.exit(1)

    # 显示操作并执行
    print(f"将重命名 {len(rename_actions)} 个文件：")
    for old, new in rename_actions:
        print(f"  {old.name} -> {new.name}")

    if args.dry_run:
        print("(dry-run模式，未实际修改)")
        return

    # 执行重命名
    for old, new in rename_actions:
        old.rename(new)
    print("重命名完成。")

if __name__ == "__main__":
    main()







