import os
import shutil
from pathlib import Path


def move_extra_files_by_name(
    ref_dir,
    check_dir,
    move_dir,
    check_exts=None,     # 只筛选 check_dir 的文件类型
    mode="move",         # "move" or "copy"
    dry_run=False        # True = 只打印不执行
):
    """
    将 check_dir 中「文件名(stem) 在 ref_dir 中不存在」的文件移动/复制到 move_dir
    不考虑扩展名，仅对文件名进行匹配
    """

    ref_dir = Path(ref_dir)
    check_dir = Path(check_dir)
    move_dir = Path(move_dir)
    move_dir.mkdir(parents=True, exist_ok=True)

    if check_exts is not None:
        if isinstance(check_exts, str):
            raise TypeError("check_exts 必须是 tuple/list，例如 ('.json',)")
        check_exts = tuple(e.lower() for e in check_exts)

    # ① 收集参考目录中的所有文件名（不看扩展名）
    ref_stems = {
        p.stem
        for p in ref_dir.iterdir()
        if p.is_file()
    }

    print(f"参考目录: {ref_dir}")
    print(f"参考文件名数量: {len(ref_stems)}")

    processed = 0

    # ② 检查目录
    for file in check_dir.iterdir():
        if not file.is_file():
            continue

        if check_exts and file.suffix.lower() not in check_exts:
            continue

        # ⭐ 核心判断：只比 stem
        if file.stem not in ref_stems:
            dst = move_dir / file.name
            action = "MOVE" if mode == "move" else "COPY"

            print(f"{action}: {file.name} -> {dst}")

            if not dry_run:
                if mode == "move":
                    shutil.move(file, dst)
                elif mode == "copy":
                    shutil.copy2(file, dst)
                else:
                    raise ValueError("mode 必须是 'move' 或 'copy'")

            processed += 1

    print(f"\n完成：处理 {processed} 个文件")

# -------------------------------
# 使用示例
# -------------------------------
if __name__ == "__main__":
    ref_dir = r"D:\DataBase\person_datas\save_datas\labels"
    check_dir = r"D:\DataBase\SpeedDifferentialGovernorDetect_Datas\train_images"
    move_dir = r"D:\Projects\Scripting_tool\output"

    move_extra_files_by_name(
        ref_dir=ref_dir,
        check_dir=check_dir,
        move_dir=move_dir,
        check_exts=(".jpg",),  # 👈 指定图像扩展名
        mode="move"             # move 或 copy
    )
