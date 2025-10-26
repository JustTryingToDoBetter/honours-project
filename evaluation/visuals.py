"""
evaluation/visuals.py
Make a side-by-side comparison video from MOT results.

Usage:
  python -c "from evaluation.visuals import render_comparison; render_comparison(
      root='/content/drive/MyDrive/fnewTrackingData/train/SNMOT-070',
      res_a='results_deepsort/deepsort_results.txt', name_a='DeepSORT',
      res_b='results_bytetrack/bytetrack_results.txt', name_b='ByteTrack',
      out='results_compare/deepsort_vs_bytetrack.mp4', fps=25
  )"
"""

import os, cv2, numpy as np, pandas as pd
from pathlib import Path

def _load_mot(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['frame','id','x','y','w','h','conf','cls','vis','_'][:len(df.columns)]
    df['frame'] = df['frame'].astype(int)
    return df

def _draw(frame, rows, color, label=None):
    for _, r in rows.iterrows():
        x,y,w,h = map(int, [r.x, r.y, r.w, r.h])
        tid = int(r.id)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{tid}", (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if label:
        cv2.rectangle(frame, (6,6), (6+160, 30), (0,0,0), -1)
        cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return frame

def render_comparison(root, res_a, name_a, res_b, name_b, out, fps=25, start_frame=None, end_frame=None,
                      metric_a=None, metric_b=None):
    root = Path(root)
    img_dir = root / "img1"
    out_dir = Path(out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    A = _load_mot(str(root / res_a))
    B = _load_mot(str(root / res_b))
    frames = sorted(set(A.frame.unique()) | set(B.frame.unique()))
    if start_frame: frames = [f for f in frames if f >= start_frame]
    if end_frame:   frames = [f for f in frames if f <= end_frame]

    # read first frame to size
    first = cv2.imread(str(img_dir / f"{frames[0]:06d}.jpg"))
    H, W = first.shape[:2]
    canvas_size = (H, W*2, 3)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W*2, H))

    for f in frames:
        img = cv2.imread(str(img_dir / f"{f:06d}.jpg"))
        if img is None: continue
        left  = img.copy()
        right = img.copy()

        rows_a = A[A.frame==f][['id','x','y','w','h']]
        rows_b = B[B.frame==f][['id','x','y','w','h']]

        left  = _draw(left,  rows_a, (0,255,0), label=name_a)
        right = _draw(right, rows_b, (0,165,255), label=name_b)

        combo = np.hstack([left, right])
        cv2.putText(combo, f"Frame {f}", (W-120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Optional metric footer
        if metric_a or metric_b:
            bar = np.zeros((36, W*2, 3), dtype=np.uint8)
            text = []
            if metric_a: text.append(f"{name_a}: MOTA {metric_a.get('mota','-')}, IDF1 {metric_a.get('idf1','-')}")
            if metric_b: text.append(f"{name_b}: MOTA {metric_b.get('mota','-')}, IDF1 {metric_b.get('idf1','-')}")
            cv2.putText(bar, " | ".join(text), (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            combo = np.vstack([combo, bar])

        writer.write(combo)

    writer.release()
    print(f"[VISUALS] Side-by-side saved → {out}")
