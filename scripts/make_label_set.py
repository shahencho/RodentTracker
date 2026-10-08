"""Sample frames from videos and pre-draw a draft mouse box on each, for review with review_labels.py.

Draft boxes come from a simple difference against the median (empty-arena) frame. They only save
labelling time; every box is checked by a person before training.

    .venv\\Scripts\\python scripts\\make_label_set.py data\\labels\\ysmu_v1 data\\ysmu\\*.mp4 --per-video 150

Split is by time block inside each video (first 70% train, next 15% val, last 15% test), so
near-identical neighbouring frames never end up on both sides.
"""
import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("videos", nargs="+")
ap.add_argument("--per-video", type=int, default=150)
ap.add_argument("--arena", default="115,70,445,415", help="x1,y1,x2,y2 region where drafts may be placed")
ap.add_argument("--pad", type=int, default=4, help="pixels added around each draft box")
args = ap.parse_args()

out = Path(args.out)
for sub in ("images", "drafts", "labels"):
    (out / sub).mkdir(parents=True, exist_ok=True)
ax1, ay1, ax2, ay2 = map(int, args.arena.split(","))
videos = [v for pattern in args.videos for v in sorted(glob.glob(pattern))]


def gray(f):
    return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def split_for(t):
    return "train" if t < 0.70 else "val" if t < 0.85 else "test"


rows = []
for video in videos:
    stem = Path(video).stem
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idxs = [int(k * (n - 1) / (args.per_video - 1)) for k in range(args.per_video)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append((i, f))
    bg = np.median(np.stack([gray(f) for _, f in frames[::2]]), axis=0).astype(np.uint8)
    bg_level = float(np.median(bg))
    found = 0
    for i, f in frames:
        name = f"{stem}_{i:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        g = gray(f).astype(np.int16)
        g = np.clip(g + int(bg_level - np.median(g)), 0, 255).astype(np.uint8)  # undo auto-exposure drift
        d = cv2.absdiff(g, bg)
        mask = np.zeros_like(d)
        mask[ay1:ay2, ax1:ax2] = 255
        _, th = cv2.threshold(cv2.bitwise_and(d, mask), 18, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        th = cv2.dilate(th, np.ones((5, 5), np.uint8))  # join head/body/tail fragments
        cnts = [c for c in cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if 40 < cv2.contourArea(c) < 4000]
        line = ""
        if cnts:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            x1, y1 = max(x - args.pad, 0), max(y - args.pad, 0)
            x2, y2 = min(x + w + args.pad, f.shape[1]), min(y + h + args.pad, f.shape[0])
            H, W = f.shape[:2]
            line = f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
            found += 1
        (out / "drafts" / f"{name}.txt").write_text(line)
        rows.append({"name": name, "video": Path(video).name, "frame": i, "time_s": round(i / fps, 2),
                     "split": split_for(i / n), "draft": int(bool(line))})
    print(f"{stem}: {len(frames)} frames, draft box on {found}")

with open(out / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} frames to {out}")
