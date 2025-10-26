"""
yolo_detect.py
===============
Optional YOLOv8 detection pipeline.
This module converts raw frames into detection text files compatible with MOT format.
"""

import os
from ultralytics import YOLO
from trackers.tracker_utils import ensure_dir
import cv2
import pandas as pd

def run_yolo_inference(img_dir, out_path, model_name='yolov8x.pt', conf=0.4):
    """
    Runs YOLOv8 detection on all images in img_dir and saves results to MOT-format .txt
    """
    model = YOLO(model_name)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))])
    results_data = []

    for i, fname in enumerate(img_files, start=1):
        path = os.path.join(img_dir, fname)
        preds = model(path, conf=conf)[0]
        for box in preds.boxes:
            x, y, w, h = box.xywh[0]
            conf_score = float(box.conf)
            cls = int(box.cls)
            results_data.append([i, -1, x, y, w, h, conf_score, cls, -1, -1])

    df = pd.DataFrame(results_data)
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, header=False, index=False)
    print(f"[YOLO Detect] Saved detections to {out_path}")
