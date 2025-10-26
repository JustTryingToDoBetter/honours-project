"""
run_bytetrack.py
================
Runs the ByteTrack tracker using pre-computed YOLO detections (MOT format)
and outputs both MOT-style tracking results and a visualization video.

Usage:
    python trackers/run_bytetrack.py
"""

import os, sys, cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from trackers.tracker_utils import load_mot_txt, draw_tracks, make_output_video, extract_pitch_mask, ensure_dir, is_valid_player

# =============================
# CONFIGURATION
# =============================
ROOT_DIR = "/content/drive/MyDrive/fnewTrackingData/train/SNMOT-070"
IMG_DIR = os.path.join(ROOT_DIR, "img1")
DET_FILE = os.path.join(ROOT_DIR, "det", "det.txt")
OUT_DIR = ensure_dir(os.path.join(ROOT_DIR, "results_bytetrack"))

FPS = 25

model = YOLO("yolov8n.pt")
# =============================
# BYTE TRACKER CONFIG
# =============================
class ByteArgs:
    track_thresh = 0.55
    match_thresh = 0.8
    track_buffer = 20
    mot20 = False

bytetrack = BYTETracker(ByteArgs(), frame_rate=FPS)

# =============================
# LOAD DETECTIONS
# =============================
det = load_mot_txt(DET_FILE)
frames = sorted(det['frame'].unique())

sample_frame = cv2.imread(os.path.join(IMG_DIR, f"{frames[0]:06d}.jpg"))
pitch_mask = extract_pitch_mask(sample_frame)

frames_out = []
results = model.track(source="OUT_DIR", tracker="bytetrack.yaml")

# =============================
# MAIN LOOP
# =============================
for f in frames:
    img_path = os.path.join(IMG_DIR, f"{f:06d}.jpg")
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    dets = det[det['frame'] == f][['x','y','w','h','conf','cls']].values
    dets = [d for d in dets if is_valid_player(d, frame.shape)]
    dets = [d for d in dets if pitch_mask is None or pitch_mask[int(d[1]+d[3]/2), int(d[0]+d[2]/2)] > 0]
    dets = np.array(dets)

    online_targets = bytetrack.update(dets, frame)
    active_tracks = []

    for t in online_targets:
        tlwh = t.tlwh
        track_id = t.track_id
        x, y, w, h = tlwh
        results.append([f, track_id, x, y, w, h, 1.0, 0, -1, -1])
        active_tracks.append({'id': track_id, 'bbox': [x, y, w, h], 'conf': 1.0})

    frame_vis = draw_tracks(frame.copy(), active_tracks, color=(0, 128, 255))
    frames_out.append(frame_vis)

print(f"[ByteTrack] Completed {len(frames)} frames.")

# =============================
# SAVE RESULTS
# =============================
res_path = os.path.join(OUT_DIR, "bytetrack_results.txt")
pd.DataFrame(results).to_csv(res_path, header=False, index=False)
print(f"[ByteTrack] Saved results to {res_path}")

vid_out_path = os.path.join(OUT_DIR, "bytetrack_output.mp4")
make_output_video(frames_out, vid_out_path, fps=FPS)
print(f"[ByteTrack] Output video saved: {vid_out_path}")
