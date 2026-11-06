"""将 NPZ 文件名写入 TXT 清单。"""

import argparse
from pathlib import Path


def write_name(npz_path, save_dir):
    """将 npz_path 目录下的 *.npz 文件 stem 写入 save_dir。"""
    source = Path(npz_path).expanduser()
    output = Path(save_dir).expanduser()
    if not source.is_dir():
        raise NotADirectoryError(f"NPZ 目录不存在：{source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.npz"), key=lambda p: p.name.lower())
    output.write_text("".join(f"{path.stem}\n" for path in files), encoding="utf-8")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_path", type=str,
        default=r"D:\DataBase\archive\dataset\semantic_drone_dataset\npz",
        help="NPZ 文件夹",
    )
    parser.add_argument(
        "--txt_file_save_dir", type=str,
        default=r"D:\DataBase\archive\dataset\semantic_drone_dataset\train.txt",
        help="输出 TXT 文件路径",
    )
    args = parser.parse_args()
    count = write_name(args.npz_path, args.txt_file_save_dir)
    print(f"完成：写入 {count} 个 NPZ 名称到 {args.txt_file_save_dir}")
