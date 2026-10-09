"""Track one mouse in an open-field video and compute the metrics.

    .venv\\Scripts\\python scripts\\analyze_video.py data\\ysmu\\video.mp4 --calib calib\\ysmu.json [--start auto|SECONDS] [--duration SECONDS]

Outputs in outputs/analysis/<video name>/:
    raw_track.csv   detector output per frame (cached; delete or pass --retrack to rerun the detector)
    track.csv       cleaned path in cm, speed, in-center flag
    metrics.csv     one row of summary metrics
    bins.csv        per-minute distance and center time
    track.png       path drawn on the video frame with the center zone
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.calibration import Calibration  # noqa: E402
from src.detector import MouseDetector  # noqa: E402
from src.metrics import center_zone, open_field  # noqa: E402
from src.tracking import auto_start_s, clean_track, track_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--calib", required=True)
ap.add_argument("--start", default="auto", help="analysis start in seconds, or 'auto' (after the hand leaves)")
ap.add_argument("--duration", type=float, help="analysis length in seconds (default: to the end)")
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--out")
ap.add_argument("--retrack", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
calib = Calibration.load(args.calib)
name = os.path.splitext(os.path.basename(args.video))[0]
out = args.out or os.path.join("outputs", "analysis", name)
os.makedirs(out, exist_ok=True)
fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)

raw_path = os.path.join(out, "raw_track.csv")
if os.path.exists(raw_path) and not args.retrack:
    raw = pd.read_csv(raw_path)
    print(f"using cached detections {raw_path}")
else:
    d = cfg["detector"]
    det = MouseDetector(d["weights"], calib, imgsz=d["imgsz"], conf=d["conf"])
    t0 = time.perf_counter()
    last = [0]

    def progress(i, total):
        if i - last[0] >= total / 10:
            last[0] = i
            el = time.perf_counter() - t0
            print(f"  {i}/{total} frames, {el / i * 1000:.0f} ms/frame, ~{(total - i) * el / i / 60:.1f} min left",
                  flush=True)

    raw = track_video(args.video, det, calib, batch=d["batch"], progress=progress)
    raw.to_csv(raw_path, index=False)
raw.attrs["fps"] = fps

tc = cfg["tracking"]
track = clean_track(raw, calib, tc["max_speed_cm_s"], tc["max_gap_s"], tc["smooth_s"])
start = auto_start_s(raw, tc["hand_foreground_px"]) if args.start == "auto" else float(args.start)
end = start + args.duration if args.duration else np.inf
win = track[(track.time_s >= start) & (track.time_s < end)].copy()
win.attrs["fps"] = fps

oc = cfg["open_field"]
summary, bins, in_center = open_field(win, calib.width_cm, calib.height_cm, oc["center_fraction"],
                                      oc["move_threshold_cm_s"], oc["min_zone_s"], oc["bin_s"])
win["in_center"] = in_center
win["speed_cm_s"] = np.hypot(win.x_cm.diff(), win.y_cm.diff()) * fps

summary = {"video": name, "arena_cm": f"{calib.width_cm:g}x{calib.height_cm:g}", **summary}
pd.DataFrame([summary]).to_csv(os.path.join(out, "metrics.csv"), index=False)
bins.to_csv(os.path.join(out, "bins.csv"), index=False)
win[["frame", "time_s", "x_px", "y_px", "conf", "detected", "jump_rejected", "x_cm", "y_cm", "speed_cm_s",
     "in_center"]].round(3).to_csv(os.path.join(out, "track.csv"), index=False)

# path plot on a frame from the middle of the analysis window
cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(win.frame.iloc[len(win) // 2]))
ok, vis = cap.read()
zx1, zy1, zx2, zy2 = center_zone(calib.width_cm, calib.height_cm, oc["center_fraction"])
cv2.polylines(vis, [calib.corners_px.astype(int)], True, (0, 0, 255), 1)
cv2.polylines(vis, [calib.to_px([[zx1, zy1], [zx2, zy1], [zx2, zy2], [zx1, zy2]]).astype(int)], True, (255, 0, 0), 1)
path = calib.to_px(win[["x_cm", "y_cm"]].dropna().to_numpy())
cv2.polylines(vis, [path.astype(np.int32)], False, (0, 160, 0), 1)
cv2.imwrite(os.path.join(out, "track.png"), vis)

print(f"\n{name}: analysed {summary['start_s']} s -> {summary['start_s'] + summary['duration_s']:.1f} s")
for k, v in summary.items():
    print(f"  {k:18s} {v}")
print(f"outputs in {out}")
