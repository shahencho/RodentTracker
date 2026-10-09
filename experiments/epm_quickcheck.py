"""Quick check of a detector on 120 frames of a Sturman EPM video against the DLC bodycentre.

Usage: .venv\\Scripts\\python experiments\\epm_quickcheck.py EPM_17 [weights]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dlcanalyzer_ref import epm_medians, read_dlc  # noqa: E402
from src.detector import MouseDetector  # noqa: E402
from src.epm import MazeCalibration  # noqa: E402

DLC_DIR = "data/dlcanalyzer/data/EPM/Output_DLC"


def dlc_file_for(video_id):
    return os.path.join(DLC_DIR, f"{video_id}DeepCut_resnet50_epmMay17shuffle1_1030000.csv")


def maze_calibration(data):
    return MazeCalibration({p: v.tolist() for p, v in epm_medians(data).items()})


if __name__ == "__main__":
    vid = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "models/mouse_det_v2.pt"
    data = read_dlc(dlc_file_for(vid))
    calib = maze_calibration(data)
    det = MouseDetector(weights, calib, imgsz=352)
    cap = cv2.VideoCapture(f"data/sturman/epm/{vid}.mp4")
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
    err, miss, tiles = [], 0, []
    for j, (i, f, d) in enumerate(zip(idxs, frames, dets)):
        ref = bc.iloc[i]
        if d is None:
            miss += 1
        elif ref.likelihood >= 0.95:
            a, b = calib.to_cm([[d[0], d[1]], [ref.x, ref.y]])
            err.append(np.hypot(*(a - b)))
        if j % 8 == 0:
            vis = f.copy()
            cv2.polylines(vis, [calib.corners_px.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(vis, (int(ref.x), int(ref.y)), 10, (0, 255, 0), 2)
            if d:
                cv2.drawMarker(vis, (int(d[0]), int(d[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
            tiles.append(cv2.resize(vis, (320, 240)))
    err = np.array(err)
    print(f"{vid}: detected {120 - miss}/120; error vs DLC bodycentre (cm): median {np.median(err):.2f}, "
          f"90th pct {np.percentile(err, 90):.2f}, >5 cm in {np.sum(err > 5)} frames")
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 5)
    cv2.imwrite(f"outputs/probe/{vid}_check.jpg", np.vstack([np.hstack(tiles[r:r + 5]) for r in range(0, len(tiles), 5)]))
