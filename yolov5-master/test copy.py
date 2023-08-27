import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import importlib.util
from tflite_runtime.interpreter import load_delegate


model_path = "runs/train/exp2/weights/best-fp16.tflite"
engine = edgetpu.detection.engine.DetectionEngine(model_path)
# model = torch.hub.load("ultralytics/yolov5", 'custom', path="./runs/train/exp2/weights/best.pt")
# model.cuda()

# print(model)D

# model cofiger
# model.score = 0.65
# model.iou = 0.45

label_dict = {
    0: "Traffic Light",
    1: "limit sign",
    2: "Stop sign",
    3: "animal",
    4: "car",
    5: "human"
}

img_path = 'test.png'

image = cv2.imread(img_path)
#result = model(path, size=640)
results = engine.detect_with_input_tensor(image, threshold=0.65, keep_aspect_ratio=True, relative_coord=False)

#bboxes = result.xyxy[0]
for obj in results:
    bbox = obj.bounding_box.flatten().tolist()
    label = obj.label
    score = obj.score
    print(f"Label: {label}, Score: {score:.2f}, Box: {bbox}")

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cls_name = label_dict[label]
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(x1, y1, x2, y2, score, cls_name)

cv2.imshow("test", image)
if cv2.waitKey(0) & 0xFF == ord('q') :
    exit()