"""
删除标签文件中的第一行
"""
import os


def remove_the_first_line(label_dir):
    label_files = os.listdir(label_dir)
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            with open(label_path, 'w') as f:
                f.writelines(lines)

if __name__ == '__main__':
    label_dir=r"D:\DataBase\WiderPerson\Annotations"
    remove_the_first_line(label_dir)

