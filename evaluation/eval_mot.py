"""
evaluation/eval_mot.py
Evaluate MOT-format predictions against MOT-format ground truth with motmetrics.

Usage:
  python evaluation/eval_mot.py \
    --root "/content/drive/MyDrive/fnewTrackingData/train/SNMOT-070" \
    --gt   "gt/gt.txt" \
    --pred "results_deepsort/deepsort_results.txt" \
    --out  "results_deepsort"

Notes:
- Expects XYWH boxes and 1-indexed frame ids. If your GT starts at 0, pass --zero_indexed_gt.
- Filters to cls==0 (person/player) by default. Change via --gt_player_cls.
"""

import os, argparse, pandas as pd, numpy as np
from pathlib import Path

# motmetrics import (with lazy install fallback)
try:
    import motmetrics as mm
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "motmetrics>=1.4.0"])
    import motmetrics as mm

def load_mot(path, zero_indexed=False):
    df = pd.read_csv(path, header=None)
    cols = ['frame','id','x','y','w','h','conf','cls','vis','_']
    df.columns = cols[:len(df.columns)]
    df['frame'] = df['frame'].astype(int)
    if zero_indexed and df['frame'].min() == 0:
        df['frame'] += 1
    return df

def tlbr_from_xywh(df):
    # motmetrics expects bounding boxes only to compute IoU via distances.iou_matrix
    a = df[['x','y','w','h']].to_numpy(dtype=float)
    tl = a[:, :2]
    br = a[:, :2] + a[:, 2:]
    return np.hstack([tl, br])

def evaluate(gt_df, pred_df, iou_thr=0.5):
    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(gt_df['frame'].unique()) | set(pred_df['frame'].unique()))
    for f in frames:
        g = gt_df[gt_df['frame']==f]
        p = pred_df[pred_df['frame']==f]

        gt_ids   = g['id'].astype(str).tolist()
        pred_ids = p['id'].astype(str).tolist()

        gt_boxes   = tlbr_from_xywh(g) if len(g) else np.empty((0,4))
        pred_boxes = tlbr_from_xywh(p) if len(p) else np.empty((0,4))

        # distance = 1 - IoU; set > (1 - thr) to np.nan so they won't match
        dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=iou_thr)
        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames','mota','motp','idf1','idp','idr','num_objects',
                 'mostly_tracked','partially_tracked','mostly_lost',
                 'num_switches','fp','fn','precision','recall'],
        name='overall'
    )
    return acc, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Sequence root folder")
    ap.add_argument("--gt",   default="gt/gt.txt")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out",  default="results_eval")
    ap.add_argument("--iou",  type=float, default=0.5)
    ap.add_argument("--zero_indexed_gt", action="store_true")
    ap.add_argument("--gt_player_cls", type=int, default=0, help="Keep GT rows with this class id")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path   = root / args.gt
    pred_path = root / args.pred

    gt   = load_mot(str(gt_path), zero_indexed=args.zero_indexed_gt)
    pred = load_mot(str(pred_path), zero_indexed=False)

    # filter to players only (cls==0 by default)
    if 'cls' in gt.columns:
        gt = gt[gt['cls'] == args.gt_player_cls].copy()

    # Drop impossible/invalid rows
    for df in (gt, pred):
        for c in ['w','h']:
            df[c] = df[c].clip(lower=0)

    _, summary = evaluate(gt, pred, iou_thr=args.iou)

    # Save CSV + readable text
    csv_path = out_dir / "metrics_summary.csv"
    summary.to_csv(csv_path)
    txt_path = out_dir / "metrics_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    print(f"[EVAL] Saved: {csv_path}")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
