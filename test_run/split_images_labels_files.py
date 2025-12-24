from FileClassification.images_labels_classification import classification


if __name__ == '__main__':
    all_file_dir = r"D:\DataBase\sdgd_datas\all_datas"

    save_map = {
        'images': {'save_path': r"D:\DataBase\sdgd_datas\images", 'suffix': ['.jpg', '.png']},
        'json_labels': {'save_path': r"D:\DataBase\sdgd_datas\jsons", 'suffix': ['.json']},
        'txt_labels': {'save_path': r"D:\DataBase\sdgd_datas\labels", 'suffix': ['.txt']}
    }

    #对文件进行分类
    classification(all_file_dir, save_map, num_workers=12, move=False)