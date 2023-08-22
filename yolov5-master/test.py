import os
import torch
import glob
import cv2

model = torch.hub.load("ultralytics/yolov5", 'custom', path="./runs/train/exp2/weights/best.pt")
model.cuda()

# print(model)D

# model cofiger
model.conf = 0.65
model.iou = 0.45

label_dict = {
    0: "Traffic Light",
    1: "limit sign",
    2: "Stop sign",
    3: "animal",
    4: "car",
    5: "human"
}

img_folder_path = "./dataset_final/test"
img_path = glob.glob(os.path.join(img_folder_path, "*", "*.jpg"))

for path in img_path:
    image = cv2.imread(path)
    result = model(path, size=640)

    bboxes = result.xyxy[0]
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        conf = conf.item()
        cls = int(cls.item())
        cls_name = label_dict[cls]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(x1, y1, x2, y2, conf, cls_name)

    cv2.imshow("test", image)
    if cv2.waitKey(0) & 0xFF == ord('q') :
        exit()