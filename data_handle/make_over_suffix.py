#修改路径下的文件后缀

import os


def make_over_suffix(input_dir,suffix):
    input_files=os.listdir(input_dir)

    for input_file in input_files:
        file_path=os.path.join(input_dir,input_file)
        os.rename(file_path,os.path.join(input_dir,os.path.splitext(input_file)[0]+suffix))




if __name__ == '__main__':
    input_dir=r"D:\DataBase\cabel_train_datas\new_datas\images"
    suffix='.jpg'
    make_over_suffix(input_dir,suffix)
