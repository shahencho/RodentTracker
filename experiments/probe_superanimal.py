"""Test DeepLabCut SuperAnimal-TopViewMouse (zero-shot) on the same 120 frames.

Run in the DeepLabCut venv after probe_yolo_candidates.py (which writes sample120.mp4):
    .venv-dlc\\Scripts\\python experiments\\probe_superanimal.py data\\ysmu\\cntrol60_cage22_an1_OF1.mp4
Licence note: SuperAnimal weights are research / non-commercial only.
"""
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import deeplabcut

sys.path.insert(0, os.path.dirname(__file__))
from probe_common import preprocess, reference_positions, sample_frames, score  # noqa: E402

video = sys.argv[1]
out = os.path.join("outputs", "probe", os.path.splitext(os.path.basename(video))[0])
frames = sample_frames(video)
refs = reference_positions(frames)

# Preprocessed copy of the sample: arena crop, 2x upscale, CLAHE
_, to_frame = preprocess(frames[0][1])
pre_path = os.path.join(out, "sample120_pre.mp4")
first, _ = preprocess(frames[0][1])
vw = cv2.VideoWriter(pre_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, first.shape[1::-1])
for _, f in frames:
    vw.write(preprocess(f)[0])
vw.release()

CONFIGS = [
    ("resnet_50", "fasterrcnn_mobilenet_v3_large_fpn"),
    ("hrnet_w32", "fasterrcnn_resnet50_fpn_v2"),
]


def centres_from_h5(h5, mapper):
    df = pd.read_hdf(h5)
    parts = df.columns.get_level_values("bodyparts").unique().tolist()
    preds = []
    for _, row in df.iterrows():
        row = row.droplevel(0)  # scorer
        if "individuals" in row.index.names:
            row = row.xs(row.index.get_level_values("individuals")[0], level="individuals")
        if "mouse_center" in parts and row[("mouse_center", "likelihood")] >= 0.1:
            x, y, c = row[("mouse_center", "x")], row[("mouse_center", "y")], row[("mouse_center", "likelihood")]
        else:
            good = [(row[(p, "x")], row[(p, "y")], row[(p, "likelihood")]) for p in parts
                    if row[(p, "likelihood")] >= 0.3]
            if not good:
                preds.append(None)
                continue
            x, y, c = np.mean([g[0] for g in good]), np.mean([g[1] for g in good]), max(g[2] for g in good)
        if np.isnan(x):
            preds.append(None)
            continue
        preds.append((*mapper(x, y), float(c)))
    return preds, parts


summary = []
for model_name, detector in CONFIGS:
    for pre in [False, True]:
        src = pre_path if pre else os.path.join(out, "sample120.mp4")
        name = f"superanimal_{model_name}{'_pre' if pre else ''}"
        dest = os.path.join(out, name)
        os.makedirs(dest, exist_ok=True)
        t = time.perf_counter()
        if glob.glob(os.path.join(dest, "*.h5")) or os.path.exists(os.path.join(out, f"{name}.csv")):
            continue  # already done
        try:
          deeplabcut.video_inference_superanimal(
            [src], "superanimal_topviewmouse", model_name=model_name, detector_name=detector,
            dest_folder=dest, max_individuals=1, bbox_threshold=0.1, pcutoff=0.1,
            video_adapt=False, device="cpu", create_labeled_video=False, plot_bboxes=False)
        except RuntimeError as e:  # DLC raises when the detector finds no mouse in any frame
            print(f"{name}: {e}")
            summary.append(score(name, frames, refs, [None] * len(frames), out))
            continue
        ms = (time.perf_counter() - t) / len(frames) * 1000
        h5 = glob.glob(os.path.join(dest, "*.h5"))[0]
        preds, parts = centres_from_h5(h5, to_frame if pre else (lambda x, y: (x, y)))
        if not summary:
            print("SuperAnimal keypoints:", parts)
        s = score(name, frames, refs, preds, out)
        s["ms_per_frame"] = round(ms)
        summary.append(s)

with open(os.path.join(out, "summary_superanimal.csv"), "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
