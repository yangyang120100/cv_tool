import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_unique_pixels(file_path):
    """读取单张mask图像，返回其唯一像素值集合"""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return set()  # 读取失败返回空集
    return set(np.unique(img))

def main():
    path = r"D:\DataBase\cabel_train_datas\aug_masks"
    # 收集所有图像文件
    image_files = []
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_files.append(os.path.join(path, filename))

    if not image_files:
        print("目录中没有图像文件")
        return

    # 设置线程数（可根据CPU核心数调整）
    max_workers = min(32, os.cpu_count() + 4)  # 合理默认值

    all_pixels = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(get_unique_pixels, f): f for f in image_files}
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_file), total=len(image_files), desc="处理中"):
            try:
                unique_set = future.result()
                all_pixels.update(unique_set)
            except Exception as e:
                file = future_to_file[future]
                print(f"处理文件 {file} 时出错: {e}")

    print("所有 mask 中出现的像素值类别：", sorted(all_pixels))

if __name__ == "__main__":
    main()