"""Quick check of a detector on 120 frames of a Sturman OFT video against the DLC bodycentre.

Usage: .venv\\Scripts\\python experiments\\sturman_quickcheck.py OFT_11 [weights]
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dlcanalyzer_ref import corner_medians, read_dlc  # noqa: E402
from src.calibration import Calibration  # noqa: E402
from src.detector import MouseDetector  # noqa: E402

LABELS = "data/dlcanalyzer/data/OFT/Labels/AllLabDataOFT_final.csv"
DLC_DIR = "data/dlcanalyzer/data/OFT/Output_DLC"


def dlc_file_for(video_id):
    lab = pd.read_csv(LABELS, sep=";")
    files = lab.loc[lab.ID == video_id, "DLCFile"].unique()
    assert len(files) == 1, (video_id, files)
    return os.path.join(DLC_DIR, files[0])


if __name__ == "__main__":
    vid = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "models/mouse_det_v1.pt"
    data = read_dlc(dlc_file_for(vid))
    calib = Calibration(corner_medians(data), 42, 42)
    det = MouseDetector(weights, calib, imgsz=352)
    cap = cv2.VideoCapture(f"data/sturman/oft/{vid}.mp4")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bc = data["bodycentre"]
    idxs = [int(100 + k * (n - 400) / 119) for k in range(120)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        frames.append(cap.read()[1])
    dets = []
    for k in range(0, 120, 16):
        dets += det.detect_batch(frames[k:k + 16])
    err, miss = [], 0
    tiles = []
    for j, (i, f, d) in enumerate(zip(idxs, frames, dets)):
        ref = bc.iloc[i]
        ok_ref = ref.likelihood >= 0.95
        if d is None:
            miss += 1
        elif ok_ref:
            a, b = calib.to_cm([[d[0], d[1]], [ref.x, ref.y]])
            err.append(np.hypot(*(a - b)))
        if j % 8 == 0:
            vis = f.copy()
            cv2.circle(vis, (int(ref.x), int(ref.y)), 6, (0, 255, 0), 2)
            if d:
                cv2.drawMarker(vis, (int(d[0]), int(d[1])), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            tiles.append(cv2.resize(vis, (310, 192)))
    err = np.array(err)
    print(f"{vid}: detected {120 - miss}/120; error vs DLC bodycentre (cm): median {np.median(err):.2f}, "
          f"90th pct {np.percentile(err, 90):.2f}, >5 cm in {np.sum(err > 5)} frames")
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 5)
    cv2.imwrite(f"outputs/probe/sturman_{vid}_check.jpg", np.vstack([np.hstack(tiles[r:r + 5]) for r in range(0, len(tiles), 5)]))
