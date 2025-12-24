import shutil
import os


def remove_name(label_dir,save_dir):
    label_files = os.listdir(label_dir)
    for label_file in label_files:
        label_name=label_file.split('.')[0]
        shutil.copy(os.path.join(label_dir, label_file),os.path.join(save_dir, label_name+'.txt'))

if __name__ == '__main__':
    label_dir=r"D:\DataBase\WiderPerson\Annotations"
    save_dir=r'D:\DataBase\WiderPerson\labels'
    remove_name(label_dir,save_dir)
