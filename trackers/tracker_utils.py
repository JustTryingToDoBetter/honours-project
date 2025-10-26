"""
tracker_utils.py
=================
Common utilities used across both DeepSORT and ByteTrack tracking scripts.
Designed for clarity, reusability, and research reproducibility.
"""

import os
import cv2
import numpy as np
import pandas as pd

# ==============================
# 1. MOT Data I/O
# ==============================

def load_mot_txt(path):
    """
    Load MOTChallenge-style text file (GT or detections).
    Expects columns: frame, id, x, y, w, h, conf, cls, vis
    """
    df = pd.read_csv(path, header=None)
    df.columns = ['frame','id','x','y','w','h','conf','cls','vis','_'][:len(df.columns)]
    df['frame'] = df['frame'].astype(int)
    df.sort_values('frame', inplace=True)
    return df


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


# ==============================
# 2. Filtering
# ==============================

def is_valid_player(b, img_shape):
    """Simple heuristic to keep only player-sized person detections."""
    H, W = img_shape[:2]
    x, y, w, h, conf, cls = b
    area = (w * h) / (W * H)
    aspect = h / max(w, 1e-6)
    return (cls == 0) and (area >= 5e-4) and (1.2 <= aspect <= 4.0)


def apply_pitch_roi(mask, boxes):
    """Keep only boxes whose center lies inside the binary pitch mask."""
    valid = []
    for b in boxes:
        x, y, w, h, conf, cls = b
        cx, cy = int(x + w/2), int(y + h/2)
        if mask is None or (0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0):
            valid.append(b)
    return valid


# ==============================
# 3. Visualization
# ==============================

def draw_tracks(frame, tracks, color=(0,255,0)):
    """
    Draw bounding boxes and IDs on a frame.
    Each track should have: track_id, bbox=[x,y,w,h]
    """
    for t in tracks:
        x, y, w, h = map(int, t['bbox'])
        track_id = int(t['id'])
        conf = t.get('conf', 1.0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"ID {track_id} ({conf:.2f})"
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def make_output_video(frames, out_path, fps=25):
    """Save list of frames to a video file."""
    if not frames:
        print("[make_output_video] No frames to save.")
        return
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()


# ==============================
# 4. Pitch ROI Mask (optional)
# ==============================

def extract_pitch_mask(frame):
    """
    Rough green-based segmentation to detect the pitch area.
    Can be improved with manual polygon if needed.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    return mask
