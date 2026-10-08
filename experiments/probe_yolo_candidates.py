"""Test KUNIN mouse-pose YOLO and YOLOE (visual prompt) on 120 frames of a video.

Usage: .venv\\Scripts\\python experiments\\probe_yolo_candidates.py data\\ysmu\\cntrol60_cage22_an1_OF1.mp4
Also writes the sampled frames as a short video + reference CSV for the DeepLabCut probe.
"""
import csv
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

sys.path.insert(0, os.path.dirname(__file__))
from probe_common import ARENA, preprocess, reference_positions, sample_frames, score  # noqa: E402

video = sys.argv[1]
out = os.path.join("outputs", "probe", os.path.splitext(os.path.basename(video))[0])
os.makedirs(out, exist_ok=True)

frames = sample_frames(video)
refs = reference_positions(frames)
with open(os.path.join(out, "reference.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "ref_x", "ref_y"])
    w.writerows([[i, *(r or ("", ""))] for (i, _), r in zip(frames, refs)])
vw = cv2.VideoWriter(os.path.join(out, "sample120.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, frames[0][1].shape[1::-1])
for _, f in frames:
    vw.write(f)
vw.release()


def in_arena(x, y):
    x1, y1, x2, y2 = ARENA
    return x1 <= x <= x2 and y1 <= y <= y2


def best_box(result, to_frame=lambda x, y: (x, y)):
    """Highest-confidence box whose centre is inside the arena, in frame coords."""
    best = None
    for b, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        cx, cy = to_frame((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if in_arena(cx, cy) and (best is None or c > best[2]):
            best = (cx, cy, c)
    return best


def run(name, predict, use_pre):
    preds = []
    t = time.perf_counter()
    for _, f in frames:
        if use_pre:
            img, to_frame = preprocess(f)
            preds.append(best_box(predict(img), to_frame))
        else:
            preds.append(best_box(predict(f)))
    ms = (time.perf_counter() - t) / len(frames) * 1000
    s = score(name, frames, refs, preds, out)
    s["ms_per_frame"] = round(ms)
    return s


summary = []

# Candidate 4: KUNIN mouse-pose YOLO (Hugging Face KUNIN/mice-yolo-inf, CC BY 4.0)
for ver in ["v0.1.4", "v0.1.5n"]:
    m = YOLO(f"models/third_party/kunin/models/kunin-mice-pose.{ver}.pt")
    if ver == "v0.1.4":
        print("KUNIN classes:", m.names, "kpt_shape:", m.model.yaml.get("kpt_shape"))
    for pre in [False, True]:
        summary.append(run(f"kunin_{ver}{'_pre' if pre else ''}",
                           lambda im: m.predict(im, conf=0.05, verbose=False)[0], pre))

# Candidate 5: YOLOE with one example box of the mouse, drawn from our own reference frame
ref_j = next(j for j, r in enumerate(refs) if r)
rx, ry = refs[ref_j]
half = 16  # ~30 px mouse
box = np.array([[rx - half, ry - half, rx + half, ry + half]])
ref_img = frames[ref_j][1]
for size in ["s", "l"]:
    m = YOLOE(f"yoloe-11{size}-seg.pt")
    for pre in [False, True]:
        if pre:
            pre_ref, _ = preprocess(ref_img)
            x1, y1 = ARENA[:2]
            vp = dict(bboxes=(box - [x1, y1, x1, y1]) * 2, cls=np.array([0]))
            r_img = pre_ref
        else:
            vp = dict(bboxes=box, cls=np.array([0]))
            r_img = ref_img
        summary.append(run(f"yoloe_11{size}_vp{'_pre' if pre else ''}",
                           lambda im: m.predict(im, visual_prompts=vp, refer_image=r_img, predictor=YOLOEVPSegPredictor,
                                                conf=0.05, verbose=False)[0], pre))

with open(os.path.join(out, "summary_yolo.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
