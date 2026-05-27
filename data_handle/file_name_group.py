#根据文件名中的字段进行分组

import shutil
from pathlib import Path
from typing import List, Optional, Literal


def group_files_by_field(
    directory: str,
    keywords: List[str],
    action: Literal["delete", "move", "copy"],
    target: Optional[str] = None,
    interactive: bool = True,
    recursive: bool = False,
    overwrite: bool = True
) -> dict:
    """
    根据文件名中最后一个下划线后的字段分组文件，并执行删除/移动/拷贝。

    参数:
        directory:  要扫描的目录路径
        keywords:   需要匹配的字段列表，如 ["color", "ins"]
        action:     操作类型，可选 "delete", "move", "copy"
        target:     移动/拷贝的目标根目录（仅 move/copy 时需要）
        interactive:若为 True，删除前请求确认，默认 True
        recursive:  若为 True，递归处理目录及子目录下所有文件，默认 False
        overwrite:  移动/拷贝时若目标已存在是否覆盖，默认 True（覆盖）

    返回:
        分组字典 {字段: [Path列表]}，便于后续检查结果

    提取规则:
        文件名 `sfhbkashfa_color.png` → 提取字段 `color`
        （取最后一个 '_' 与扩展名之间的部分）
    """
    src_dir = Path(directory)
    if not src_dir.is_dir():
        raise NotADirectoryError(f"{src_dir} 不是有效目录")

    if action in ("move", "copy") and target is None:
        raise ValueError("move/copy 操作必须提供 target 参数")

    target_dir = Path(target) if target else None

    # 1. 提取字段
    def extract_field(filepath: Path) -> str:
        stem = filepath.stem
        if '_' not in stem:
            return ''
        return stem.rsplit('_', 1)[1]

    # 2. 根据递归模式收集文件
    groups: dict = {kw: [] for kw in keywords}
    if recursive:
        file_iterator = (p for p in src_dir.rglob('*') if p.is_file())
    else:
        file_iterator = (p for p in src_dir.iterdir() if p.is_file())

    for fpath in file_iterator:
        field = extract_field(fpath)
        if field in keywords:
            groups[field].append(fpath)

    total = sum(len(lst) for lst in groups.values())
    if total == 0:
        print("未找到任何匹配的文件。")
        return groups

    print(f"找到 {total} 个匹配文件，分组情况：")
    for kw, files in groups.items():
        print(f"  {kw}: {len(files)} 个")

    # 3. 删除确认
    if action == "delete" and interactive:
        confirm = input("确认要永久删除以上文件？(yes/no): ")
        if confirm.lower() != "yes":
            print("操作已取消。")
            return groups

    # 4. 执行操作
    for field, files in groups.items():
        print(f"处理字段 '{field}' ({len(files)} 个文件)")
        for src in files:
            try:
                if action == "delete":
                    src.unlink()
                    print(f"已删除: {src.name}")
                else:
                    dest_dir = target_dir / field
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / src.name

                    if dest.exists():
                        if overwrite:
                            dest.unlink()
                            print(f"覆盖已存在文件: {dest.name}")
                        else:
                            print(f"跳过已存在文件: {dest.name}")
                            continue

                    if action == "move":
                        shutil.move(str(src), str(dest))
                        print(f"已移动: {src.name} → {dest}")
                    else:  # copy
                        shutil.copy2(str(src), str(dest))
                        print(f"已拷贝: {src.name} → {dest}")
            except Exception as e:
                print(f"处理 {src.name} 失败: {e}")

    print("操作完成。")
    return groups

if __name__ == '__main__':
    group_files_by_field(r"D:\DataBase\road\goose_2d_train\images\train",['vis'],'copy',r"D:\DataBase\road\goose\train\images",recursive=True)
