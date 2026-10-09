"""Check the product engine (src: homography calibration + zone masks + open_field) against the DLCAnalyzer answer key.

For each Sturman test video, the DLC bodycentre track goes through the reference port (dlcanalyzer_ref) and our
cached detector track goes through src.metrics.open_field, both on the same analysis window. This is the number a
user of the UI would see versus the published-method number.

Usage: .venv\\Scripts\\python experiments\\engine_vs_reference.py [models\\mouse_det_v2.pt]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dlcanalyzer_ref import clean_point, corner_medians, oft_metrics, read_dlc  # noqa: E402
from sturman_quickcheck import dlc_file_for  # noqa: E402
from src.calibration import Calibration  # noqa: E402
from src.metrics import open_field  # noqa: E402
from src.zones import ZoneSpec  # noqa: E402

FPS, START, END = 25, 100, 250
TRAIN = {"OFT_5", "OFT_44", "OFT_6", "OFT_12", "OFT_23", "OFT_16"}  # as in validate_sturman.py
TEST = sorted({l.split()[0][:-4] for l in open("data/sturman/oft_urls.txt")} - TRAIN, key=lambda v: int(v[4:]))
model = os.path.splitext(os.path.basename(sys.argv[1] if len(sys.argv) > 1 else "mouse_det_v2"))[0]
rows = []
for vid in TEST:
    data = read_dlc(dlc_file_for(vid))
    corners = corner_medians(data)
    ref = oft_metrics(clean_point(data["bodycentre"]), corners)

    raw = pd.read_csv(f"outputs/validation/{model}/{vid}_raw.csv")
    xy_px = np.c_[raw.x_px.interpolate(limit_direction="both"), raw.y_px.interpolate(limit_direction="both")]
    n = len(xy_px)
    keep = np.arange(n)[START:]
    keep = keep[:len(keep) - (END + 1)]
    calib = Calibration(corners, 42, 42)
    s, table, _, _ = open_field(calib.to_cm(xy_px[keep]), keep / FPS, FPS, ZoneSpec(42, 42))
    rows.append({"video": vid, "ref_distance": ref["raw_distance_cm"], "ours_distance": s["total_distance_cm"],
                 "ref_centre_s": ref["center_time_s"], "ours_centre_s": table.loc["centre", "time_s"],
                 "ref_moving_s": ref["time_moving_s"], "ours_moving_s": s["time_moving_s"]})

df = pd.DataFrame(rows)
df.to_csv(f"outputs/validation/{model}/engine_vs_reference.csv", index=False)
print(df.round(1).to_string(index=False))
for a, b in [("ours_distance", "ref_distance"), ("ours_centre_s", "ref_centre_s"), ("ours_moving_s", "ref_moving_s")]:
    d = df[a] - df[b]
    print(f"{a:15s} mean abs diff {d.abs().mean():7.1f}  mean abs % {(d.abs() / df[b] * 100).mean():5.1f}%  "
          f"max abs % {(d.abs() / df[b] * 100).max():5.1f}%  r = {np.corrcoef(df[a], df[b])[0, 1]:.3f}")
