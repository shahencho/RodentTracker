"""Replay the box choice on saved candidates with different settings and score each against the EPM reference.

Needs <video>_candidates.csv from validate_epm.py. Runs in seconds, so the choice rules can be tuned without
re-running the detector. Usage: .venv\\Scripts\\python experiments\\epm_replay.py EPM_17 [EPM_19 ...]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dlcanalyzer_ref import epm_medians, epm_reference, read_dlc  # noqa: E402
from epm_quickcheck import dlc_file_for  # noqa: E402
from src.epm import MazeCalibration  # noqa: E402
from src.tracking import replay  # noqa: E402

FPS = 25
OUT = "outputs/validation/mouse_det_v2_epm"
GRID = {"fg_weight": [0.0, 0.5, 1.0], "far_penalty": [0.0, 0.3, 1.0]}

rows = []
for vid in sys.argv[1:]:
    data = read_dlc(dlc_file_for(vid))
    med = epm_medians(data)
    calib = MazeCalibration({p: v.tolist() for p, v in med.items()})
    cands = pd.read_csv(os.path.join(OUT, f"{vid}_candidates.csv"))
    bc = data["bodycentre"]
    n = min(len(bc), int(cands.frame.max()) + 1)
    ref = epm_reference(bc[["x", "y"]].to_numpy()[:n], bc.likelihood.to_numpy()[:n], med)
    scale = 65.5 / np.hypot(*(med["tl"] - med["br"]))
    good_ref = bc.likelihood.to_numpy()[:n] >= 0.95
    for fw, fp in itertools.product(GRID["fg_weight"], GRID["far_penalty"]):
        xyc = replay(cands, range(n), FPS, calib, fg_weight=fw, far_penalty=fp)
        ours = epm_reference(xyc[:, :2], np.where(np.isnan(xyc[:, 0]), 0.0, 1.0), med)
        ok = good_ref & ~np.isnan(xyc[:, 0])
        err = np.hypot(*(xyc[ok, :2] - bc[["x", "y"]].to_numpy()[:n][ok]).T) * scale
        rows.append({"video": vid, "fg_weight": fw, "far_penalty": fp,
                     "open_diff_s": ours["open_time_s"] - ref["open_time_s"],
                     "closed_diff_s": ours["closed_time_s"] - ref["closed_time_s"],
                     "centre_diff_s": ours["center_time_s"] - ref["center_time_s"],
                     "distance_diff_pct": 100 * (ours["raw_distance_cm"] / ref["raw_distance_cm"] - 1),
                     "off5cm_pct": 100 * float(np.mean(err > 5))})

df = pd.DataFrame(rows)
print(df.groupby(["fg_weight", "far_penalty"])[["open_diff_s", "closed_diff_s", "centre_diff_s",
                                                "distance_diff_pct", "off5cm_pct"]]
      .agg(lambda v: np.mean(np.abs(v))).round(2).to_string())
