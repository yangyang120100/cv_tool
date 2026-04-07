import os
import re
import sys

def has_chinese(text):
    """判断字符串是否包含中文字符（CJK统一表意文字）"""
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 基本汉字范围
    return bool(pattern.search(text))

def delete_chinese_files(root_dir, dry_run=True):
    """
    递归删除文件名包含中文的文件
    :param root_dir: 根目录路径
    :param dry_run: 若为True，只打印而不实际删除；为False则执行删除
    """
    deleted_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if has_chinese(filename):
                file_path = os.path.join(dirpath, filename)
                if dry_run:
                    print(f"[DRY RUN] 将删除: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"已删除: {file_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除失败 {file_path}: {e}")
    if dry_run:
        print(f"\n[DRY RUN] 共发现 {deleted_count} 个待删除文件。")
    else:
        print(f"\n实际删除了 {deleted_count} 个文件。")

if __name__ == "__main__":
    # 请修改为你的目标根目录
    target_dir = r"D:\DataBase\Insulator_datas\jsons"

    if not os.path.exists(target_dir):
        print(f"错误：目录不存在 - {target_dir}")
        sys.exit(1)

    print("递归删除文件名包含中文的文件")
    print("当前模式：仅预览（不会真正删除）")
    print("若要实际删除，请将 dry_run=False 传入函数。")
    print("-" * 50)

    # 先预览（dry_run=True）
    delete_chinese_files(target_dir, dry_run=True)

    # 询问是否继续实际删除
    answer = input("\n是否要实际删除上述文件？(yes/no): ").strip().lower()
    if answer == "yes":
        print("\n开始实际删除...")
        delete_chinese_files(target_dir, dry_run=False)
    else:
        print("操作已取消，未删除任何文件。")