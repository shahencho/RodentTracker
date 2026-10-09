"""Fill frames YOLOE missed (flag no_box) with a box from background subtraction.

Writes the image + label next to the YOLOE ones and a fills.csv, plus a contact sheet
of every filled box for checking by eye.
Usage: .venv\\Scripts\\python experiments\\fill_misses_bgsub.py data\\ysmu\\*.mp4 [--calib FILE_OR_DIR] [--min-side 14]
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe_common import ARENA, sample_frames  # noqa: E402
from src.calibration import Calibration  # noqa: E402

LABELS = os.path.join("data", "labels", "yoloe")


def blob_box(frame, bg, arena, min_side):
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    shift = int(np.median(bg)) - int(np.median(g))  # undo auto-exposure drift
    g = cv2.add(g, shift) if shift > 0 else cv2.subtract(g, -shift)
    x1, y1, x2, y2 = arena
    mask = np.zeros_like(bg)
    mask[y1:y2, x1:x2] = 255
    d = cv2.bitwise_and(cv2.absdiff(g, bg), mask)
    _, th = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # join body parts split by low contrast
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if 40 < cv2.contourArea(c) < 3000]
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if min(w, h) < min_side:  # only part of the mouse separated from the background
        return None
    return x, y, x + w, y + h


ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--calib")
ap.add_argument("--min-side", type=int, default=14, help="px; a whole mouse is never thinner than this")
args = ap.parse_args()

for video in args.videos:
    name = os.path.splitext(os.path.basename(video))[0]
    arena = ARENA
    if args.calib:
        path = os.path.join(args.calib, name + ".json") if os.path.isdir(args.calib) else args.calib
        cap0 = cv2.VideoCapture(video)
        shape = (int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)))
        arena = Calibration.load(path).crop_box(shape, 0.08)
    d = os.path.join(LABELS, name)
    missed = [int(r["frame"]) for r in csv.DictReader(open(os.path.join(d, "detections.csv"))) if r["flag"] == "no_box"]
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for _, f in sample_frames(video, 60)]
    bg = np.median(np.stack(gray), axis=0).astype(np.uint8)
    cap = cv2.VideoCapture(video)
    rows, tiles = [], []
    for idx in missed:
        stale = os.path.join(d, "images", f"{name}_{idx:06d}.jpg")  # from an earlier run of this script
        for p in (stale, stale.replace("images", "labels").replace(".jpg", ".txt")):
            if os.path.exists(p):
                os.remove(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        box = blob_box(f, bg, arena, args.min_side) if ok else None
        rows.append([idx, *(box or ("", "", "", "")), "bgsub" if box else "unfilled"])
        vis = f.copy()
        if box:
            x1, y1, x2, y2 = box
            stem = f"{name}_{idx:06d}"
            cv2.imwrite(os.path.join(d, "images", stem + ".jpg"), f)
            h, w = f.shape[:2]
            with open(os.path.join(d, "labels", stem + ".txt"), "w") as fh:
                fh.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vis = vis[max(0, cy - 60):cy + 60, max(0, cx - 60):cx + 60]
        tiles.append(cv2.resize(vis, (160, 160)))
        cv2.putText(tiles[-1], str(idx), (3, 14), 0, 0.4, (0, 0, 255), 1)
    with open(os.path.join(d, "fills.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "x1", "y1", "x2", "y2", "source"])
        wr.writerows(rows)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
        cv2.imwrite(os.path.join("outputs", "probe", f"{name}_fills.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    print(f"{name}: {len(missed)} missed, {sum(r[-1] == 'bgsub' for r in rows)} filled from background subtraction")
