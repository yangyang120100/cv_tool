import json
import argparse
import sys
from pathlib import Path
from glob import glob
import os


def remove_labels_from_data(data, labels_to_remove, label_field="label"):
    """从 shapes 数组中删除指定 label 的 shape，返回修改后的 data 和删除数量"""
    if not isinstance(data, dict) or "shapes" not in data:
        raise ValueError("JSON 顶层必须是包含 'shapes' 字段的对象")
    shapes = data["shapes"]
    if not isinstance(shapes, list):
        raise ValueError("'shapes' 字段必须为列表")

    remove_set = set(labels_to_remove)
    original_count = len(shapes)
    filtered = [s for s in shapes if s.get(label_field) not in remove_set]
    data["shapes"] = filtered
    removed = original_count - len(filtered)
    return data, removed


def process_single_file(input_path, output_path, labels, label_field, indent):
    """处理单个文件，返回删除的 shape 数量，出错返回 -1"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [错误] 读取失败 {input_path}: {e}", file=sys.stderr)
        return -1

    try:
        data, removed = remove_labels_from_data(data, labels, label_field)
    except Exception as e:
        print(f"  [错误] 处理失败 {input_path}: {e}", file=sys.stderr)
        return -1

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    except Exception as e:
        print(f"  [错误] 写入失败 {output_path}: {e}", file=sys.stderr)
        return -1

    return removed


def main():
    parser = argparse.ArgumentParser(
        description="从图像标注JSON的shapes中批量删除指定 label 的标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单文件处理
  python remove_label_batch.py input.json output.json -l woods

  # 多个文件
  python remove_label_batch.py a.json b.json c.json -o out/ -l dirt water

  # 目录批量处理（默认 *.json）
  python remove_label_batch.py --input-dir ./labels -o clean_labels/ -l woods

  # 目录 + 自定义匹配模式
  python remove_label_batch.py --input-dir ./data --pattern "*.labelme.json" -l woods --inplace

  # 原地覆盖（谨慎！）
  python remove_label_batch.py input.json -l woods --inplace
        """
    )

    # 输入源：可以是多个文件路径，或 --input-dir
    parser.add_argument("files", nargs="*", help="输入 JSON 文件（可多个）")
    parser.add_argument("--input-dir", "-d", help="输入目录，批量处理目录中的文件",default=r"E:\add_road_06_24\jsons")
    parser.add_argument("--pattern", default="*.json", help="与 --input-dir 配合的文件匹配模式 (默认: *.json)")

    # 输出选项
    parser.add_argument("-o", "--output-dir", help="输出目录（不指定时需与 --inplace 配合）")
    parser.add_argument("--inplace", default=True,action="store_true", help="原地覆盖输入文件（危险操作，建议先备份）")

    # 删除标签
    parser.add_argument("-l", "--labels", nargs="+", default=['conductor'], help="要删除的 label，可多个空格分隔")
    parser.add_argument("--label-field", default="label", help="shapes 中表示类别的字段名 (默认: label)")

    # 其他
    parser.add_argument("--indent", type=int, default=2, help="输出 JSON 缩进空格数 (默认: 2)")

    args = parser.parse_args()

    # 收集要处理的文件列表
    input_files = []

    if args.input_dir:
        # 从目录匹配
        base_dir = Path(args.input_dir)
        if not base_dir.is_dir():
            print(f"错误: 输入目录不存在 - {base_dir}", file=sys.stderr)
            sys.exit(1)
        pattern_path = base_dir / args.pattern
        matched = [Path(p) for p in glob(str(pattern_path))]
        if not matched:
            print(f"警告: 在 {base_dir} 中没有匹配 '{args.pattern}' 的文件")
        input_files.extend(matched)

    # 加上直接列出的文件
    for f in args.files:
        p = Path(f)
        if p.is_file():
            input_files.append(p)
        else:
            print(f"警告: 忽略无效文件 {f}")

    if not input_files:
        print("错误: 没有指定任何输入文件。请通过文件列表或 --input-dir 指定。", file=sys.stderr)
        sys.exit(1)

    # 确定输出逻辑
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    if not out_dir and not args.inplace:
        print("错误: 必须指定 --output-dir 或使用 --inplace 进行原地覆盖。", file=sys.stderr)
        sys.exit(1)

    if out_dir and args.inplace:
        print("提示: 同时指定了 --output-dir 和 --inplace，将优先使用 --output-dir 输出到指定目录。")

    # 处理每个文件
    total_files = len(input_files)
    total_removed = 0
    success_count = 0

    for idx, input_path in enumerate(input_files, 1):
        print(f"处理 [{idx}/{total_files}] {input_path}")

        # 决定输出路径
        if args.output_dir:
            # 保持相对目录结构？此处只保留文件名
            output_path = out_dir / input_path.name
        else:
            # inplace 模式
            output_path = input_path

        removed = process_single_file(input_path, output_path, args.labels, args.label_field, args.indent)
        if removed >= 0:
            total_removed += removed
            success_count += 1
            print(f"  -> 已删除 {removed} 个 shape，保存到 {output_path}")
        else:
            print(f"  -> 处理失败，已跳过")

    print(f"\n完成！成功处理 {success_count}/{total_files} 个文件，共删除 {total_removed} 个标注。")


if __name__ == "__main__":
    main()