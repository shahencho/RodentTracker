"""Validate the plus-maze analysis on the Sturman EPM videos against the DLCAnalyzer answer key.

For each video (none was used to train the detector):
  ref     DLC bodycentre track through the DLCAnalyzer EPM port (the answer key)
  method  our detector track through the same port (same maze points, cut, cleaning, zones)
  engine  our product engine (12-point homography, our zones, src.epm measures) on the same time window
plus the per-frame distance between our point and the DLC bodycentre. Detector output is cached.

Usage: .venv\\Scripts\\python experiments\\validate_epm.py models\\mouse_det_v2.pt [--videos EPM_1 ...]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dlcanalyzer_ref import epm_medians, epm_reference, read_dlc  # noqa: E402
from epm_quickcheck import dlc_file_for  # noqa: E402
from src.analysis import Setup, analyze  # noqa: E402
from src.detector import MouseDetector  # noqa: E402
from src.epm import MazeCalibration  # noqa: E402
from src.tracking import track_video  # noqa: E402

FPS, START, END = 25, 100, 250
ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--videos", nargs="+")
args = ap.parse_args()
videos = args.videos or sorted((os.path.basename(p)[:-4] for p in glob.glob("data/sturman/epm/EPM_*.mp4")),
                               key=lambda v: int(v[4:]))
cfg = yaml.safe_load(open("config.yaml"))
model = os.path.splitext(os.path.basename(args.weights))[0]
out = os.path.join("outputs", "validation", f"{model}_epm")
os.makedirs(out, exist_ok=True)

rows = []
for vid in videos:
    data = read_dlc(dlc_file_for(vid))
    med = epm_medians(data)
    points = {p: v.tolist() for p, v in med.items()}
    calib = MazeCalibration(points)
    raw_path = os.path.join(out, f"{vid}_raw.csv")
    if not os.path.exists(raw_path):
        t0 = time.perf_counter()
        try:
            raw = track_video(f"data/sturman/epm/{vid}.mp4", MouseDetector(args.weights, calib), calib,
                              candidates_csv=os.path.join(out, f"{vid}_candidates.csv"))
        except Exception as e:  # e.g. a video still downloading
            print(f"{vid}: skipped ({e})", flush=True)
            continue
        raw.to_csv(raw_path, index=False)
        print(f"{vid}: tracked {len(raw)} frames in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    raw = pd.read_csv(raw_path)

    bc = data["bodycentre"]
    n = min(len(raw), len(bc))
    ref = epm_reference(bc[["x", "y"]].to_numpy()[:n], bc.likelihood.to_numpy()[:n], med)
    ours_xy = raw[["x_px", "y_px"]].to_numpy()[:n]
    method = epm_reference(ours_xy, np.where(np.isnan(ours_xy[:, 0]), 0.0, 1.0), med)

    setup = Setup(test="epm", maze_mode="corners", maze_points_px=points,
                  start_s=START / FPS, end_s=(n - END - 2) / FPS)
    eng = analyze(raw.iloc[:n], FPS, setup, cfg)
    zt, s = eng["zones"], eng["summary"]

    scale = 65.5 / np.hypot(*(med["tl"] - med["br"]))
    good = (bc.likelihood.to_numpy()[:n] >= 0.95) & ~np.isnan(ours_xy[:, 0])
    err = np.hypot(*(ours_xy[good] - bc[["x", "y"]].to_numpy()[:n][good]).T) * scale

    row = {"video": vid, "detected_pct": round(100 * float(raw.x_px.notna().mean()), 1),
           "err_median_cm": round(float(np.median(err)), 2), "frames_err_gt5cm_pct": round(100 * float(np.mean(err > 5)), 2)}
    for k in ["open_time_s", "closed_time_s", "center_time_s", "open_transitions", "closed_transitions",
              "raw_distance_cm", "distance_moving_cm"]:
        row[f"ref_{k}"], row[f"ours_{k}"] = round(ref[k], 1), round(method[k], 1)
    row["engine_open_time_s"] = zt.loc["open", "time_s"]
    row["engine_closed_time_s"] = zt.loc["closed", "time_s"]
    row["engine_centre_time_s"] = zt.loc["centre", "time_s"]
    row["engine_open_entries"] = zt.loc["open", "entries"]
    row["engine_distance_cm"] = s["total_distance_cm"]
    row["ref_open_time_pct"] = round(100 * ref["open_time_s"] / (ref["open_time_s"] + ref["closed_time_s"]), 1)
    row["engine_open_time_pct"] = s["open_time_pct"]
    rows.append(row)
    print(f"{vid}: open arms ours {method['open_time_s']:.1f} / engine {row['engine_open_time_s']:.1f} vs ref "
          f"{ref['open_time_s']:.1f} s; distance ours {method['raw_distance_cm']:.0f} vs ref {ref['raw_distance_cm']:.0f} cm; "
          f"median error {row['err_median_cm']} cm", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "per_video.csv"), index=False)


def report(a, b, label):
    d = df[a] - df[b]
    pct = (d.abs() / df[b].replace(0, np.nan) * 100)
    print(f"{label:34s} mean abs diff {d.abs().mean():7.1f}  bias {d.mean():+7.1f}  mean abs % {pct.mean():5.1f}%  "
          f"r = {np.corrcoef(df[a], df[b])[0, 1]:.3f}")


print(f"\n=== {model} on {len(df)} EPM videos ===")
for k in ["open_time_s", "closed_time_s", "center_time_s", "open_transitions", "raw_distance_cm", "distance_moving_cm"]:
    report(f"ours_{k}", f"ref_{k}", f"method: {k}")
report("engine_open_time_s", "ref_open_time_s", "engine: open_time_s")
report("engine_closed_time_s", "ref_closed_time_s", "engine: closed_time_s")
report("engine_centre_time_s", "ref_center_time_s", "engine: centre_time_s")
report("engine_open_time_pct", "ref_open_time_pct", "engine: % time in open arms")
report("engine_distance_cm", "ref_raw_distance_cm", "engine: distance")
print(f"per-frame error: median {df.err_median_cm.median():.2f} cm, frames > 5 cm off {df.frames_err_gt5cm_pct.mean():.2f}%, "
      f"detected {df.detected_pct.mean():.1f}%")
