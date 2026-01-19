from FileClassification.images_labels_classification import classification


if __name__ == '__main__':
    all_file_dir = r"D:\DataBase\cabel_train_datas\add_cable_datas\datas"

    save_map = {
        'images': {'save_path':r'D:\DataBase\cabel_train_datas\add_cable_datas\images', 'suffix': ['.jpg', '.png',".JPG"]},
        'json_labels': {'save_path': r"D:\DataBase\cabel_train_datas\add_cable_datas\jsons", 'suffix': ['.json']},
        'txt_labels': {'save_path': r"D:\DataBase\cabel_train_datas\add_cable_datas\labels", 'suffix': ['.txt']}
    }

    #对文件进行分类
    classification(all_file_dir, save_map, num_workers=12, move=False)