"""Track one mouse in an open-field video and compute the metrics (command-line version of the UI).

    .venv\\Scripts\\python scripts\\analyze_video.py data\\ysmu\\video.mp4 --calib calib\\ysmu_cage22.json [--start S] [--end S]

Outputs in outputs/analysis/<video name>/:
    raw_track.csv   detector output per frame (cached; delete or pass --retrack to rerun the detector)
    results.csv     one row: settings, whole-test and zone measures
    track.csv       per-frame position (px and cm), speed, moving and zone flags
    bins.csv        per-minute distance, time moving and centre time
    track.png, heatmap.png, per_minute.png, report.pdf
"""
import argparse
import os
import sys
import time

import cv2
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import report  # noqa: E402
from src.analysis import Setup, analyze  # noqa: E402
from src.calibration import Calibration  # noqa: E402
from src.detector import MouseDetector  # noqa: E402
from src.tracking import track_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--calib", required=True, help="calibration JSON (corners + arena size)")
ap.add_argument("--start", type=float, help="analysis start in seconds (default: automatic)")
ap.add_argument("--end", type=float, help="analysis end in seconds (default: automatic)")
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--out")
ap.add_argument("--retrack", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
calib = Calibration.load(args.calib)
d = cfg["defaults"]
setup = Setup(calib.corners_px.tolist(), calib.width_cm, calib.height_cm, d["grid_cm"], d["center_mode"],
              d["center_value"], start_s=args.start, end_s=args.end)
name = os.path.splitext(os.path.basename(args.video))[0]
out = args.out or os.path.join("outputs", "analysis", name)
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)

raw_path = os.path.join(out, "raw_track.csv")
if os.path.exists(raw_path) and not args.retrack:
    raw = pd.read_csv(raw_path)
    print(f"using cached detections {raw_path}")
else:
    dc = cfg["detector"]
    det = MouseDetector(dc["weights"], calib, imgsz=dc["imgsz"], conf=dc["conf"])
    t0 = time.perf_counter()
    last = [0]

    def progress(i, total):
        if i - last[0] >= total / 10:
            last[0] = i
            el = time.perf_counter() - t0
            print(f"  {i}/{total} frames, {el / i * 1000:.0f} ms/frame, ~{(total - i) * el / i / 60:.1f} min left",
                  flush=True)

    raw = track_video(args.video, det, calib, batch=dc["batch"], progress=progress)
    raw.to_csv(raw_path, index=False)

result = analyze(raw, fps, setup, cfg)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(result["per_frame"].frame.iloc[len(result["per_frame"]) // 2]))
frame = cap.read()[1]

with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
    fh.write(report.summary_csv(name, setup, result))
with open(os.path.join(out, "track.csv"), "w", newline="") as fh:
    fh.write(report.track_csv(result))
result["per_bin"].to_csv(os.path.join(out, "bins.csv"), index=False)
for fname, data in [("track.png", report.png(report.track_figure(frame, result))),
                    ("heatmap.png", report.png(report.heatmap_figure(result))),
                    ("per_minute.png", report.png(report.per_minute_figure(result))),
                    ("report.pdf", report.pdf_bytes(name, setup, frame, result))]:
    with open(os.path.join(out, fname), "wb") as fh:
        fh.write(data)

whole, header, zone_rows = report.measures_tables(result)
print(f"\n{name}  (auto start {result['auto_start_s']} s, auto end {result['auto_end_s']} s)")
for k, v in whole:
    print(f"  {k:26s} {v}")
print("  " + " | ".join(header))
for r in zone_rows:
    print("  " + " | ".join(r))
print(f"outputs in {out}")
