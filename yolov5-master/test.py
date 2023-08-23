import os
#import torch
import glob
import cv2
import edgetpu.detection.engine

model_path = "yolov5-master\runs\train\exp2\weights\best.pb"
engine = edgetpu.detection.engine.DetectionEngine(model_path)
# model = torch.hub.load("ultralytics/yolov5", 'custom', path="./runs/train/exp2/weights/best.pt")
# model.cuda()

# print(model)D

# model cofiger
# model.conf = 0.65
# model.iou = 0.45

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
    #result = model(path, size=640)
    results = engine.detect_with_input_tensor(image, threshold=0.65, keep_aspect_ratio=True, relative_coord=False)

    #bboxes = result.xyxy[0]
    for obj in results:
        bbox = obj.bounding_box.flatten().tolist()
        class_id = obj.label
        conf = obj.score

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_name = label_dict[class_id]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(x1, y1, x2, y2, conf, cls_name)

    cv2.imshow("test", image)
    if cv2.waitKey(0) & 0xFF == ord('q') :
        exit()