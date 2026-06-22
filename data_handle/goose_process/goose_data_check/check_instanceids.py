#全景mask标签编码转码输出检查
import json
import cv2
import numpy as np

with open(r"C:\Users\Kedio\Desktop\panoptic_val.json") as f:
    data = json.load(f)

for i in range(len(data['annotations'])):
    ann = data["annotations"][i]
    car_ids = []

    for seg in ann["segments_info"]:

        if seg["category_id"] == 12:  # car
            car_ids.append(seg["id"])
    if len(car_ids)>0:
        print(ann["file_name"])
        print("car segments:", len(car_ids))
        print(car_ids[:20])