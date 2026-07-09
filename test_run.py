path = r"E:\MyRoad\classes.txt"

classes_map = dict()
with open(path, encoding='utf-8') as w:
    t_datas = w.readlines()
    count = 0
    for t in t_datas:
        t = t.strip()          # 去除换行符及两端空白
        # 或者 t = t.rstrip('\n')   # 仅去除末尾换行符
        if t:                  # 可选：跳过空行
            classes_map[t] = count
            count += 1
print(classes_map)
print(classes_map.keys())
# from pathlib import Path
#
# image_dir=r"E:\MyRoad\images"
# mask_dir=r"E:\MyRoad\masks"
#
# for image_name in os.listdir(image_dir):
#     image_path=os.path.join(image_dir,image_name)
#     mask_path=os.path.join(mask_dir,Path(image_name).stem+".png")
#     if not os.path.exists(mask_path):
#         print(image_name)
#         print(mask_path)
#         print('\n')
#     # image_shape=cv2.imread(image_path).shape[0:2]
#     # mask_shape=cv2.imread(mask_path).shape[0:2]
#     # if image_shape!=mask_shape:
#     #     print(image_name)

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 1. 读取图像（你已有的代码）
# img = cv2.imread(r"E:\test_jyz\images\DJI_20241220152847_0010_V.JPG")
# if img is None:
#     raise FileNotFoundError("图像读取失败，请检查路径")
#
# # 转为灰度图，大部分纹理/特征提取都在灰度上进行
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 2. Sobel 梯度幅值（突出纹理边缘的方向变化）
# sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
# sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))
#
# # 3. Canny 边缘检测（精细的边缘轮廓）
# canny = cv2.Canny(gray, 50, 150)
#
# # 4. Laplacian 算子（二阶导数，检测灰度突变区域）
# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# laplacian = np.uint8(np.absolute(laplacian))
#
# # 5. Gabor 滤波（多方向、多尺度纹理响应）
# def create_gabor_response(gray_img, theta=0, sigma=4.0, lambd=10.0, gamma=0.5):
#     kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
#     filtered = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
#     return filtered
#
# gabor_0   = create_gabor_response(gray, theta=0)
# gabor_45  = create_gabor_response(gray, theta=np.pi/4)
# gabor_90  = create_gabor_response(gray, theta=np.pi/2)
# gabor_135 = create_gabor_response(gray, theta=3*np.pi/4)
#
# # 6. Harris 角点检测（角点作为显著特征点）
# harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# # 在原图上绘制角点
# img_harris = img.copy()
# img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]  # 红色标记
#
# # 7. Shi-Tomasi 角点检测（Good Features to Track）
# corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
# img_shi = img.copy()
# if corners is not None:
#     for c in corners.astype(int):
#         cv2.circle(img_shi, (c[0][0], c[0][1]), 5, (0, 255, 0), -1)  # 绿色圆点
#
# # 8. 可视化：使用 matplotlib 展示多个子图
# plt.figure(figsize=(16, 12))
#
# plt.subplot(3, 3, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# plt.axis("off")
#
# plt.subplot(3, 3, 2)
# plt.imshow(sobel_mag, cmap='gray')
# plt.title("Sobel Magnitude")
# plt.axis("off")
#
# plt.subplot(3, 3, 3)
# plt.imshow(canny, cmap='gray')
# plt.title("Canny Edges")
# plt.axis("off")
#
# plt.subplot(3, 3, 4)
# plt.imshow(laplacian, cmap='gray')
# plt.title("Laplacian")
# plt.axis("off")
#
# plt.subplot(3, 3, 5)
# plt.imshow(gabor_0, cmap='gray')
# plt.title("Gabor (theta=0°)")
# plt.axis("off")
#
# plt.subplot(3, 3, 6)
# plt.imshow(gabor_45, cmap='gray')
# plt.title("Gabor (theta=45°)")
# plt.axis("off")
#
# plt.subplot(3, 3, 7)
# plt.imshow(gabor_90, cmap='gray')
# plt.title("Gabor (theta=90°)")
# plt.axis("off")
#
# plt.subplot(3, 3, 8)
# plt.imshow(gabor_135, cmap='gray')
# plt.title("Gabor (theta=135°)")
# plt.axis("off")
#
# plt.subplot(3, 3, 9)
# plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
# plt.title("Harris Corners")
# plt.axis("off")
#
# plt.tight_layout()
# plt.savefig("texture_features.jpg")
# plt.show()

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ---------- 配置 ----------
# image_path = r"E:\test_jyz\images\DJI_20241220152847_0010_V.JPG"
# output_dir = "feature_output"          # 保存特征图的文件夹
# os.makedirs(output_dir, exist_ok=True) # 自动创建
#
# # 1. 读取图像
# img = cv2.imread(image_path)
# if img is None:
#     raise FileNotFoundError("图像读取失败，请检查路径")
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # ---------- 2. Sobel 梯度幅值 ----------
# sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
# sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))
# cv2.imwrite(os.path.join(output_dir, "sobel_magnitude.png"), sobel_mag)
#
# # ---------- 3. Canny 边缘 ----------
# canny = cv2.Canny(gray, 50, 150)
# cv2.imwrite(os.path.join(output_dir, "canny_edges.png"), canny)
#
# # ---------- 4. Laplacian ----------
# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# laplacian = np.uint8(np.absolute(laplacian))
# cv2.imwrite(os.path.join(output_dir, "laplacian.png"), laplacian)
#
# # ---------- 5. Gabor 滤波 ----------
# def create_gabor_response(gray_img, theta=0, sigma=4.0, lambd=10.0, gamma=0.5):
#     # 修正：生成浮点型核，滤波后保持浮点，再归一化保存
#     kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
#     # 使用CV_32F输出，避免类型错误
#     filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
#     # 归一化到0-255以便保存
#     filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
#     return np.uint8(filtered)
#
# gabor_0   = create_gabor_response(gray, theta=0)
# gabor_45  = create_gabor_response(gray, theta=np.pi/4)
# gabor_90  = create_gabor_response(gray, theta=np.pi/2)
# gabor_135 = create_gabor_response(gray, theta=3*np.pi/4)
#
# cv2.imwrite(os.path.join(output_dir, "gabor_0deg.png"), gabor_0)
# cv2.imwrite(os.path.join(output_dir, "gabor_45deg.png"), gabor_45)
# cv2.imwrite(os.path.join(output_dir, "gabor_90deg.png"), gabor_90)
# cv2.imwrite(os.path.join(output_dir, "gabor_135deg.png"), gabor_135)
#
# # ---------- 6. Harris 角点标记图 ----------
# harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# img_harris = img.copy()
# img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]  # 红色标记
# cv2.imwrite(os.path.join(output_dir, "harris_corners.png"), img_harris)
#
# # ---------- 7. Shi-Tomasi 角点标记图 ----------
# corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
# img_shi = img.copy()
# if corners is not None:
#     for c in corners.astype(int):
#         cv2.circle(img_shi, (c[0][0], c[0][1]), 5, (0, 255, 0), -1)  # 绿色
# cv2.imwrite(os.path.join(output_dir, "shi_tomasi_corners.png"), img_shi)
#
# # ---------- 可选：保存原图 ----------
# cv2.imwrite(os.path.join(output_dir, "original.png"), img)
#
# # ---------- 8. 可视化（保留原显示） ----------
# plt.figure(figsize=(16, 12))