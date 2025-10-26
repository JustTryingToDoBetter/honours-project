"""
run_deepsort.py
================
Runs the DeepSORT tracker using pre-computed YOLO detections (MOT format)
and outputs both MOT-style tracking results and a visualization video.

Usage:
    python trackers/run_deepsort.py
"""

import os, sys, cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from trackers.tracker_utils import load_mot_txt, draw_tracks, make_output_video, extract_pitch_mask, ensure_dir, is_valid_player

# =============================
# CONFIGURATION
# =============================
ROOT_DIR = "/content/drive/MyDrive/fnewTrackingData/train/SNMOT-070"
IMG_DIR = os.path.join(ROOT_DIR, "img1")
DET_FILE = os.path.join(ROOT_DIR, "det", "det.txt")
OUT_DIR = ensure_dir(os.path.join(ROOT_DIR, "results_deepsort"))

CONF_THRESH = 0.4
FPS = 25

# =============================
# INITIALIZE TRACKER
# =============================
deepsort = DeepSort(
    max_age=20,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=50,
    embedder="mobilenet",
)

# =============================
# LOAD DETECTIONS
# =============================
det = load_mot_txt(DET_FILE)
frames = sorted(det['frame'].unique())

# Sample one frame to build pitch mask
sample_frame = cv2.imread(os.path.join(IMG_DIR, f"{frames[0]:06d}.jpg"))
pitch_mask = extract_pitch_mask(sample_frame)

frames_out = []
results = []

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

    tracks = deepsort.update_tracks(dets, frame=frame)
    active_tracks = []

    for t in tracks:
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        x, y, w, h = t.to_ltrb()
        track_data = [f, t.track_id, x, y, w-x, h-y, 1.0, 0, -1, -1]
        results.append(track_data)
        active_tracks.append({'id': t.track_id, 'bbox': [x, y, w-x, h-y], 'conf': 1.0})

    frame_vis = draw_tracks(frame.copy(), active_tracks)
    frames_out.append(frame_vis)

print(f"[DeepSORT] Completed {len(frames)} frames.")

# =============================
# SAVE RESULTS
# =============================
res_path = os.path.join(OUT_DIR, "deepsort_results.txt")
pd.DataFrame(results).to_csv(res_path, header=False, index=False)
print(f"[DeepSORT] Saved results to {res_path}")

vid_out_path = os.path.join(OUT_DIR, "deepsort_output.mp4")
make_output_video(frames_out, vid_out_path, fps=FPS)
print(f"[DeepSORT] Output video saved: {vid_out_path}")
