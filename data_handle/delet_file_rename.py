import os
import glob


def remove_substring_from_filename(file_path, substring):
    """
    参数:
        file_path (str): 文件完整路径
        substring (str): 要删除的字段，例如 "labelids"

    返回:
        bool: 操作成功返回 True，否则返回 False
    """
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return False

    # 分离目录和文件名
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    # 如果文件名中不包含要删除的字段，则跳过
    if substring not in base_name:
        print(f"跳过：文件名中未找到子串 '{substring}' - {base_name}")
        return False

    # 删除所有出现的子串（若只需删除第一个，可使用 replace(substring, '', 1)）
    new_base_name = base_name.replace(substring, '')
    new_path = os.path.join(dir_name, new_base_name)

    # 检查新文件名是否已存在，避免覆盖
    if os.path.exists(new_path):
        print(f"错误：目标文件已存在，无法重命名 - {new_path}")
        return False

    try:
        os.rename(file_path, new_path)
        print(f"成功：{file_path} -> {new_path}")
        return True
    except Exception as e:
        print(f"重命名失败：{e}")
        return False


def batch_remove_substring(directory, substring, pattern="*"):
    """
    批量处理目录下符合模式的文件，删除文件名中的指定子串。

    参数:
        directory (str): 目标目录路径
        substring (str): 要删除的字段
        pattern (str):  文件名匹配模式，如 "*.png" 或 "*labelids*"，默认为所有文件
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"在目录 '{directory}' 中未找到匹配 '{pattern}' 的文件")
        return

    for file_path in files:
        remove_substring_from_filename(file_path, substring)


# ========== 使用示例 ==========
if __name__ == "__main__":
    # # 示例1：处理单个文件
    # single_file = r"C:\path\to\2022-07-22_flight__0000_1658492967230070008_labelids.png"
    # remove_substring_from_filename(single_file, "labelids")

    # 示例2：批量处理某目录下所有 .png 文件
    batch_remove_substring(r"D:\DataBase\road\goose\val\images\vis", "_windshield_vis", "*.png")