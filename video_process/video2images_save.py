import os
import threading
from pathlib import Path
import cv2
"""
将视频转换为图像序列并保存
"""
__all__=['video2images_and_save']

def video2images(video_path,save_dir,save_fps):
    cap = cv2.VideoCapture(video_path)
    video_pather=Path(video_path)
    video_name=video_pather.stem
    image_count=0

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break
        if image_count%save_fps==0:
            cv2.imwrite(os.path.join(save_dir, f"{video_name}_{image_count}.jpg"), frame)
        image_count += 1

    cap.release()
    cv2.destroyAllWindows()


def video2images_and_save(video_dir,save_image_dir,save_fps):
    video_file_names=os.listdir(video_dir)

    for video_file_name in video_file_names:
        video_path=os.path.join(video_dir,video_file_name)
        video2image_thread=threading.Thread(target=video2images,args=(video_path,save_image_dir,save_fps))
        # video2images(video_path,save_image_dir)
        video2image_thread.start()


# if __name__ == '__main__':
    # video_dir=r"D:\SpeedDifferentialGovernorDetect_Datas\train_videos"
    # save_image_dir=r"D:\SpeedDifferentialGovernorDetect_Datas\train_images"
    # video2images_and_save(video_dir,save_image_dir)