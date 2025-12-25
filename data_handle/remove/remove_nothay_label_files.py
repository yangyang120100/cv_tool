"""
删除空标签文件
"""
import os

# 要扫描的目录
directory = r"D:\Projects\ultralytics\output\scs_body\labels"  # 改成你自己的路径

# 遍历目录及子目录
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            # 判断文件是否为空（大小为0 或 内容为空）
            if os.path.getsize(file_path) == 0:
                print(f"Deleting empty file: {file_path}")
                os.remove(file_path)
            else:
                # 如果想进一步判断文件内容是否全是空白字符
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"Deleting empty file: {file_path}")
                        os.remove(file_path)
