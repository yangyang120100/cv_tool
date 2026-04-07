import os

# 修改为你图片的实际绝对路径
target_dir = r"/home/yy/DataBase/insulator_datas_coco/val2017"


def batch_rename(path):
    count = 0
    for filename in os.listdir(path):
        # 检查是否以小写 .jpg 结尾
        if filename.endswith(".jpg"):
            old_path = os.path.join(path, filename)
            # 生成新文件名
            new_name = filename[:-4] + ".JPG"
            new_path = os.path.join(path, new_name)

            # 执行重命名
            os.rename(old_path, new_path)
            print(f"成功: {filename} -> {new_name}")
            count += 1

    print(f"\n✨ 处理完成，共重命名 {count} 个文件。")


if __name__ == "__main__":
    if os.path.exists(target_dir):
        batch_rename(target_dir)
    else:
        print("❌ 错误：找不到指定的路径！")