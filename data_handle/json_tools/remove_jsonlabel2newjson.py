import json
import argparse
import sys
from pathlib import Path


def merge_labels(base_data, add_data, target_labels, label_field="label"):
    """
    将 add_data['shapes'] 中 label 属于 target_labels 的项
    追加到 base_data['shapes'] 末尾，返回合并后的 base_data 和追加数量。
    """
    if not isinstance(base_data, dict) or "shapes" not in base_data:
        raise ValueError("基础 JSON 必须为包含 'shapes' 字段的对象")
    if not isinstance(add_data, dict) or "shapes" not in add_data:
        raise ValueError("增加 JSON 必须为包含 'shapes' 字段的对象")

    base_shapes = base_data["shapes"]
    add_shapes = add_data["shapes"]
    if not isinstance(base_shapes, list) or not isinstance(add_shapes, list):
        raise ValueError("'shapes' 字段必须为列表")

    target_set = set(target_labels)
    to_add = [s for s in add_shapes if s.get(label_field) in target_set]
    base_shapes.extend(to_add)
    return base_data, len(to_add)


def get_json_files(directory, pattern="*.json"):
    """获取目录下所有 JSON 文件（不递归），返回 {文件名: 完整路径} 的字典"""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    files = {}
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            files[file_path.name] = file_path
    return files


def main():
    parser = argparse.ArgumentParser(
        description="将增加目录中指定 label 的 shape 合并到同名基础 JSON 中",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 输出到新目录
  python merge_shapes.py --base-dir ./base --add-dir ./add -l woods dirt -o merged/

  # 原地覆盖基础文件（谨慎！）
  python merge_shapes.py --base-dir ./base --add-dir ./add -l concrete-pavement --inplace
        """
    )

    parser.add_argument("--base-dir", default=r"E:\add_road_06_24\jsons", help="基础 JSON 文件所在目录")
    parser.add_argument("--add-dir", default=r"E:\add_road_06_24\jsonsold", help="需要从中提取 shape 的 JSON 目录")
    parser.add_argument("-l", "--labels", nargs="+", default=['conductor'], help="要提取的 label 类别，可多个")
    parser.add_argument("--label-field", default="label", help="label 字段名 (默认: label)")
    parser.add_argument("-o", "--output-dir", help="输出目录（不指定时需搭配 --inplace）")
    parser.add_argument("--inplace",default=True, action="store_true", help="原地覆盖基础目录中的文件")
    parser.add_argument("--indent", type=int, default=2, help="输出 JSON 缩进 (默认 2)")

    args = parser.parse_args()

    # 确定输出方式
    if not args.output_dir and not args.inplace:
        print("错误: 必须指定 --output-dir 或使用 --inplace 进行原地覆盖。", file=sys.stderr)
        sys.exit(1)
    if args.output_dir and args.inplace:
        print("提示: 同时指定了 --output-dir 和 --inplace，将优先输出到指定目录。")

    try:
        base_files = get_json_files(args.base_dir)
        add_files = get_json_files(args.add_dir)
    except Exception as e:
        print(f"读取目录失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not base_files:
        print("基础目录中没有找到 JSON 文件。")
    if not add_files:
        print("增加目录中没有找到 JSON 文件。")

    # 找出同名文件
    common_names = set(base_files.keys()) & set(add_files.keys())
    if not common_names:
        print("两个目录中没有同名的 JSON 文件，无需处理。")
        sys.exit(0)

    # 确定输出目录
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None  # inplace 模式

    total_files = len(common_names)
    total_added = 0
    success_count = 0

    for idx, fname in enumerate(sorted(common_names), 1):
        print(f"处理 [{idx}/{total_files}] {fname}")
        base_path = base_files[fname]
        add_path = add_files[fname]

        try:
            with open(base_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
            with open(add_path, 'r', encoding='utf-8') as f:
                add_data = json.load(f)
        except Exception as e:
            print(f"  [错误] 读取文件失败: {e}")
            continue

        try:
            merged_data, added = merge_labels(base_data, add_data, args.labels, args.label_field)
        except Exception as e:
            print(f"  [错误] 合并数据失败: {e}")
            continue

        # 输出路径
        if out_dir:
            output_path = out_dir / fname
        else:
            output_path = base_path

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=args.indent)
            print(f"  -> 已追加 {added} 个 shape，保存到 {output_path}")
            total_added += added
            success_count += 1
        except Exception as e:
            print(f"  [错误] 写入文件失败: {e}")

    print(f"\n完成！成功处理 {success_count}/{total_files} 个文件，共追加 {total_added} 个标注。")


if __name__ == "__main__":
    main()