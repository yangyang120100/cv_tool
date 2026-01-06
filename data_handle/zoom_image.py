#图像缩放 并保存原始图像信息
import cv2

def resize_keep_ratio(image, target_size, by_width=True):
    """
    按指定边长缩放图像，保持原宽高比
    :param image: 原始图像
    :param target_size: 目标边长（宽或高）
    :param by_width: 是否按宽缩放，True为按宽，False为按高
    :return: 缩放后的图像, 原始宽高
    """
    original_height, original_width = image.shape[:2]
    if by_width:
        new_width = target_size
        new_height = int(original_height * (target_size / original_width))
    else:
        new_height = target_size
        new_width = int(original_width * (target_size / original_height))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, (original_width, original_height)

def restore_to_original_size(resized_image, original_size):
    """
    将缩放后的图像恢复到原始分辨率
    :param resized_image: 缩放后的图像
    :param original_size: 原始宽高 (width, height)
    :return: 恢复到原始分辨率的图像
    """
    original_width, original_height = original_size
    restored_image = cv2.resize(resized_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return restored_image

if __name__ == '__main__':
    img = cv2.imread(r"D:\Projects\Scripting_tool\test_data\images\DJI_20250509123811_0124_V.JPG")
    # 按宽缩放到400像素
    resized_img, original_size = resize_keep_ratio(img, 2048, by_width=True)
    print(f"原始尺寸: {original_size}, 缩放后尺寸: {resized_img.shape[1]}x{resized_img.shape[0]}")

    cv2.imwrite(r"D:\Projects\Scripting_tool\test_data\images\DJI_20250509123811_0124_V.JPG", resized_img)
    # cv2.imshow('resize_img_by_width', resized_img)
    # cv2.waitKey(0)
    #
    # # 按高缩放到300像素
    # resized_img2, _ = resize_keep_ratio(img, 300, by_width=False)
    # print(f"按高缩放后尺寸: {resized_img2.shape[1]}x{resized_img2.shape[0]}")
    # cv2.imshow('resize_img_by_height', resized_img2)
    # cv2.waitKey(0)
    #
    # # 恢复到原始分辨率
    # restored_img = restore_to_original_size(resized_img, original_size)
    # cv2.imshow('restored_img', restored_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
