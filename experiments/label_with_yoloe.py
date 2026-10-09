"""Auto-label mouse boxes with YOLOE (4-example visual prompt) for fine-tuning a small detector.

For each video: build the prompt from 4 frames spread over the arena (positions from the
background-subtraction reference), embed it once, then run every STEP-th frame.
Writes YOLO-format images/labels plus detections.csv; frames with no box or a suspicious
jump are listed for hand-checking.

Usage: .venv\\Scripts\\python experiments\\label_with_yoloe.py data\\ysmu\\*.mp4
       .venv\\Scripts\\python experiments\\label_with_yoloe.py data\\sturman\\oft\\OFT_5.mp4 --calib calib\\sturman --half 22 --step 25
--calib: a calibration JSON, or a folder of <video name>.json; without it the YSMU arena box is used.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe_common import ARENA, HALF, best_box, build_reference, pick_examples, reference_positions, sample_frames  # noqa: E402
from src.calibration import Calibration  # noqa: E402

CONF = 0.01
N_EXAMPLES = 4
JUMP_PX = 80  # flag if the mouse moves further than this between labelled frames
OUT_ROOT = os.path.join("data", "labels", "yoloe")

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--calib")
ap.add_argument("--half", type=int, default=HALF, help="half size of the example boxes (px)")
ap.add_argument("--max-size", type=int, default=90, help="larger boxes are not a mouse (px)")
ap.add_argument("--step", type=int, default=5, help="label every STEP-th frame")
args = ap.parse_args()


def arena_for(video, name):
    """Label region: the calibrated floor plus a small margin, or the YSMU box."""
    if not args.calib:
        return ARENA
    path = os.path.join(args.calib, name + ".json") if os.path.isdir(args.calib) else args.calib
    cap = cv2.VideoCapture(video)
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    return Calibration.load(path).crop_box(shape, pad_fraction=0.08)


for video in args.videos:
    name = os.path.splitext(os.path.basename(video))[0]
    arena = arena_for(video, name)
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)

    frames = sample_frames(video)
    refs = reference_positions(frames, arena)
    examples = pick_examples(refs, N_EXAMPLES, arena)
    ref_img, boxes = build_reference(frames, refs, examples, arena, args.half)
    cv2.imwrite(os.path.join(out, "prompt.jpg"), ref_img)

    m = YOLOE("yoloe-11l-seg.pt")
    m.predict(ref_img, refer_image=ref_img, visual_prompts=dict(bboxes=boxes, cls=np.zeros(len(boxes), int)),
              predictor=YOLOEVPSegPredictor, verbose=False)

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, prev, n_box, n_flag = [], None, 0, 0
    t0 = time.perf_counter()
    idx = -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.step:
            continue
        b = best_box(m.predict(f, conf=CONF, verbose=False)[0], args.max_size, arena)
        if b is None:
            rows.append([idx, "", "", "", "", "", "no_box"])
            n_flag += 1
            continue
        cx, cy, c, (x1, y1, x2, y2) = b
        flag = "jump" if prev and np.hypot(cx - prev[0], cy - prev[1]) > JUMP_PX else ""
        n_flag += bool(flag)
        prev = (cx, cy)
        rows.append([idx, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(c, 3), flag])
        stem = f"{name}_{idx:06d}"
        cv2.imwrite(os.path.join(out, "images", stem + ".jpg"), f)
        h, w = f.shape[:2]
        with open(os.path.join(out, "labels", stem + ".txt"), "w") as fh:
            fh.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        n_box += 1
        if len(rows) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"{name}: {idx}/{total} frames, {n_box} boxes, {n_flag} flagged, "
                  f"{el / len(rows):.2f} s/frame, ~{(total - idx) / args.step * el / len(rows) / 60:.0f} min left", flush=True)

    with open(os.path.join(out, "detections.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "flag"])
        wr.writerows(rows)
    print(f"{name}: DONE {len(rows)} frames sampled, {n_box} boxes, {n_flag} flagged for hand-check", flush=True)
