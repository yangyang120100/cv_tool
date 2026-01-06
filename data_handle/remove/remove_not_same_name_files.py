import shutil
from pathlib import Path
from typing import Callable


def default_stem_mapper(stem: str) -> str:
    """
    将 mask / label 文件名映射为 image stem
    例如:
        0001_mask -> 0001
        IMG_01_label -> IMG_01
    """
    for suffix in ("_mask", "_label", "_gt", "_seg"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def move_extra_files_by_name(
    ref_dir,
    check_dir,
    move_dir=None,
    check_exts=None,
    on_mismatch="move",
    dry_run=False,
    recursive=True,
    stem_mapper: Callable[[str], str] = None,
):
    """
    处理 check_dir 中「stem(映射后) 不在 ref_dir 中」的文件

    on_mismatch:
        - "move"   : 移动到 move_dir
        - "copy"   : 复制到 move_dir
        - "delete" : 删除
        - "ignore" : 只打印
    """

    valid_actions = {"move", "copy", "delete", "ignore"}
    if on_mismatch not in valid_actions:
        raise ValueError(f"on_mismatch 必须是 {valid_actions}")

    if stem_mapper is None:
        stem_mapper = default_stem_mapper

    ref_dir = Path(ref_dir)
    check_dir = Path(check_dir)

    if on_mismatch in {"move", "copy"}:
        if move_dir is None:
            raise ValueError("on_mismatch 为 move/copy 时必须指定 move_dir")
        move_dir = Path(move_dir)
        move_dir.mkdir(parents=True, exist_ok=True)

    if check_exts is not None:
        if isinstance(check_exts, str):
            raise TypeError("check_exts 必须是 tuple/list，例如 ('.png',)")
        check_exts = tuple(e.lower() for e in check_exts)

    # ---------- ① 收集参考 stem ----------
    ref_files = ref_dir.rglob("*") if recursive else ref_dir.iterdir()
    ref_stems = {
        stem_mapper(p.stem)
        for p in ref_files
        if p.is_file()
    }

    print(f"参考目录: {ref_dir}")
    print(f"参考 stem 数量: {len(ref_stems)}")

    processed = 0

    # ---------- ② 扫描 check_dir ----------
    check_files = check_dir.rglob("*") if recursive else check_dir.iterdir()

    for file in check_files:
        if not file.is_file():
            continue

        if check_exts and file.suffix.lower() not in check_exts:
            continue

        mapped_stem = stem_mapper(file.stem)

        if mapped_stem in ref_stems:
            continue

        # ---------- 不匹配文件处理 ----------
        action = on_mismatch.upper()

        if on_mismatch in {"move", "copy"}:
            dst = move_dir / file.name
            if dst.exists():
                dst = move_dir / f"{file.stem}_dup{file.suffix}"

            print(f"{action}: {file} -> {dst}")

            if  dry_run:
                if on_mismatch == "move":
                    shutil.move(str(file), str(dst))
                else:
                    shutil.copy2(str(file), str(dst))

        elif on_mismatch == "delete":
            if  dry_run:
                file.unlink()
                print(f"DELETE: {file}")

        elif on_mismatch == "ignore":
            print(f"IGNORE: {file}")

        processed += 1

    print(f"\n完成：处理 {processed} 个不匹配文件")


if __name__ == '__main__':
    move_extra_files_by_name(
        ref_dir=r"D:\DataBase\cabel_train_datas\augmented_wash\val\wash_images",#参考路径
        check_dir=r"D:\DataBase\cabel_train_datas\train_datas\val\wash_masks",#被检查的路径
        check_exts=(".png",),#检查的后缀名
        on_mismatch="delete",#不匹配时的处理方式
        dry_run=True
    )