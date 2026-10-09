"""Score a detector on the 120 YSMU probe frames per video (same harness and reference as the v1 numbers).

Usage: .venv\Scripts\python experiments\ysmu_regression.py models\mouse_det_v2.pt
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe_common import reference_positions, sample_frames, score  # noqa: E402
from src.calibration import Calibration  # noqa: E402
from src.detector import MouseDetector  # noqa: E402

weights = sys.argv[1]
tag = os.path.splitext(os.path.basename(weights))[0]
det = MouseDetector(weights, Calibration.load("calib/ysmu_cage22.json"))
for video in sorted(glob.glob("data/ysmu/*.mp4")):
    frames = sample_frames(video)
    refs = reference_positions(frames)
    preds = []
    for k in range(0, len(frames), 16):
        preds += det.detect_batch([f for _, f in frames[k:k + 16]])
    score(f"{tag}_ysmu", frames, refs, preds, os.path.join("outputs", "probe", os.path.splitext(os.path.basename(video))[0]))
