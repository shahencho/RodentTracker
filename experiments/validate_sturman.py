"""Validate our tracker on the Sturman open-field videos against the DLCAnalyzer answer key.

For each video: run our detector on every frame, then compute the DLCAnalyzer OFT metrics (exact port, same
corners, same cut, same zones) both on our track and on the DLC 'bodycentre' track. Differences therefore come
from tracking only. Also reports the per-frame distance between our point and the DLC bodycentre.

Usage: .venv\\Scripts\\python experiments\\validate_sturman.py models\\mouse_det_v2.pt [--videos OFT_11 OFT_14 ...]
Default videos: the 14 not used for training. Detector output is cached per model in outputs/validation/.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dlcanalyzer_ref import clean_point, corner_medians, oft_metrics, polygon_area, read_dlc  # noqa: E402
from sturman_quickcheck import dlc_file_for  # noqa: E402
from src.calibration import Calibration  # noqa: E402
from src.detector import MouseDetector  # noqa: E402
from src.tracking import track_video  # noqa: E402

TRAIN = {"OFT_5", "OFT_44", "OFT_6", "OFT_12", "OFT_23", "OFT_16"}
ALL = [l.split()[0][:-4] for l in open("data/sturman/oft_urls.txt")]
TEST = sorted(set(ALL) - TRAIN, key=lambda s: int(s[4:]))

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--videos", nargs="+", default=TEST)
args = ap.parse_args()

model_name = os.path.splitext(os.path.basename(args.weights))[0]
out = os.path.join("outputs", "validation", model_name)
os.makedirs(out, exist_ok=True)
rows = []
for vid in args.videos:
    data = read_dlc(dlc_file_for(vid))
    corners = corner_medians(data)
    calib = Calibration(corners, 42, 42)
    raw_path = os.path.join(out, f"{vid}_raw.csv")
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
    else:
        t0 = time.perf_counter()
        raw = track_video(f"data/sturman/oft/{vid}.mp4", MouseDetector(args.weights, calib), calib)
        raw.to_csv(raw_path, index=False)
        print(f"{vid}: tracked {len(raw)} frames in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    # our track, gaps filled the same way DLCAnalyzer fills low-likelihood points
    ours = np.c_[raw.x_px.interpolate(limit_direction="both"), raw.y_px.interpolate(limit_direction="both")]
    ref_xy = clean_point(data["bodycentre"])
    n = min(len(ours), len(ref_xy))
    ref = oft_metrics(ref_xy[:n], corners)
    mine = oft_metrics(ours[:n], corners)

    # per-frame position error (frames where DLC is confident), in cm with DLCAnalyzer's scale
    px_to_cm = np.sqrt(42 * 42 / polygon_area(corners))
    bc = data["bodycentre"].iloc[:n]
    good = (bc.likelihood >= 0.95).to_numpy() & raw.x_px.iloc[:n].notna().to_numpy()
    err = np.hypot(raw.x_px.iloc[:n] - bc.x, raw.y_px.iloc[:n] - bc.y).to_numpy()[good] * px_to_cm

    row = {"video": vid, "detected_pct": round(100 * raw.x_px.notna().mean(), 1),
           "err_median_cm": round(float(np.median(err)), 2), "err_p95_cm": round(float(np.percentile(err, 95)), 2),
           "frames_err_gt5cm_pct": round(100 * float(np.mean(err > 5)), 2)}
    for k in ["raw_distance_cm", "distance_moving_cm", "time_moving_s", "center_time_s", "center_transitions"]:
        row[f"ref_{k}"] = round(ref[k], 1)
        row[f"ours_{k}"] = round(mine[k], 1)
    rows.append(row)
    print(f"{vid}: distance ours {mine['raw_distance_cm']:.0f} vs ref {ref['raw_distance_cm']:.0f} cm; "
          f"center ours {mine['center_time_s']:.1f} vs ref {ref['center_time_s']:.1f} s; "
          f"median error {row['err_median_cm']} cm", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "per_video.csv"), index=False)
print(f"\n=== {model_name} on {len(df)} videos ===")
for k in ["raw_distance_cm", "distance_moving_cm", "time_moving_s", "center_time_s", "center_transitions"]:
    a, b = df[f"ours_{k}"], df[f"ref_{k}"]
    pct = (a - b) / b.replace(0, np.nan) * 100
    print(f"{k:20s} mean abs diff {np.mean(np.abs(a - b)):8.1f}  bias {np.mean(a - b):+8.1f}  "
          f"mean abs % {np.nanmean(np.abs(pct)):5.1f}%  r = {np.corrcoef(a, b)[0, 1]:.3f}")
print(f"per-frame error: median of medians {df.err_median_cm.median():.2f} cm, "
      f"frames > 5 cm off {df.frames_err_gt5cm_pct.mean():.2f}%, detected {df.detected_pct.mean():.1f}%")
