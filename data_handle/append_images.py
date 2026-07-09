import cv2
import numpy as np

def hconcat_images_cv(image_paths, output_path):
    """
    将多张图像水平拼接，使用 OpenCV。
    所有图像高度会被调整为一致（以第一张图的高度为基准）。
    """
    images = [cv2.imread(p) for p in image_paths]
    if any(img is None for img in images):
        raise FileNotFoundError("有图像读取失败，请检查路径。")

    # 统一高度
    base_height = images[0].shape[0]
    resized_images = []
    for img in images:
        if img.shape[0] != base_height:
            ratio = base_height / img.shape[0]
            new_width = int(img.shape[1] * ratio)
            img = cv2.resize(img, (new_width, base_height), interpolation=cv2.INTER_LANCZOS4)
        resized_images.append(img)

    # 水平拼接
    result = np.hstack(resized_images)  # 等效于 cv2.hconcat
    cv2.imwrite(output_path, result)
    print(f"拼接完成，保存至：{output_path}")

# 示例调用
if __name__ == "__main__":
    hconcat_images_cv([r"D:\PythonProject\cv_tool\feature_output\canny_edges.png",
                       r"D:\PythonProject\cv_tool\feature_output\laplacian.png",
                       r"D:\PythonProject\cv_tool\feature_output\sobel_magnitude.png"], "hconcat_cv.jpg")