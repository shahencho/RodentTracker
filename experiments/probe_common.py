"""Shared harness for testing off-the-shelf mouse detectors on one video.

Samples N evenly spaced frames, builds a reference mouse position per frame with a
background-subtraction prototype (checked by eye on the contact sheet), and scores
candidate detections against it.
"""
import csv
import os

import cv2
import numpy as np

N_FRAMES = 120
HIT_RADIUS_PX = 30
HALF = 16  # half example-box size for a ~30 px mouse
MAX_FOREGROUND_PX = 2500  # a mouse is ~300-1500 px of foreground; more means a hand is in the arena
# Arena interior of the YSMU sample (x1, y1, x2, y2), hand-measured; replaced by clicked corners later
ARENA = (110, 65, 450, 420)


def sample_frames(video, n=N_FRAMES):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in range(n):
        idx = int(k * (total - 1) / (n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            out.append((idx, f))
    return out


def preprocess(frame, scale=2):
    """Crop to the arena, upscale and boost contrast (CLAHE). Returns image and a mapper back to frame coords."""
    x1, y1, x2, y2 = ARENA
    crop = frame[y1:y2, x1:x2]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def to_frame(x, y):
        return x / scale + x1, y / scale + y1

    return img, to_frame


def reference_positions(frames, arena=ARENA, max_fg=MAX_FOREGROUND_PX):
    """Mouse centre per frame from median-background subtraction (None if no blob)."""
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for _, f in frames]
    bg = np.median(np.stack(gray[::2]), axis=0).astype(np.uint8)
    x1, y1, x2, y2 = arena
    mask = np.zeros_like(bg)
    mask[y1:y2, x1:x2] = 255
    refs = []
    for g in gray:
        shift = int(np.median(bg)) - int(np.median(g))  # undo auto-exposure drift
        g = cv2.add(g, shift) if shift > 0 else cv2.subtract(g, -shift)
        d = cv2.bitwise_and(cv2.absdiff(g, bg), mask)
        _, th = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(th) > max_fg:  # experimenter's hand in view: no reliable mouse position
            refs.append(None)
            continue
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if 40 < cv2.contourArea(c) < 3000]
        if not cnts:
            refs.append(None)
            continue
        m = cv2.moments(max(cnts, key=cv2.contourArea))
        refs.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return refs


def median_background(video, n=60):
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for _, f in sample_frames(video, n)]
    return np.median(np.stack(gray), axis=0).astype(np.uint8)


def hand_in_view(frame, bg, arena=ARENA, max_fg=MAX_FOREGROUND_PX):
    """True when far more of the arena differs from the background than a mouse could explain."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    shift = int(np.median(bg)) - int(np.median(g))
    g = cv2.add(g, shift) if shift > 0 else cv2.subtract(g, -shift)
    x1, y1, x2, y2 = arena
    d = cv2.absdiff(g, bg)[y1:y2, x1:x2]
    return cv2.countNonZero(cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)[1]) > max_fg


def score(name, frames, refs, preds, outdir):
    """preds: list of (cx, cy, conf) or None per frame, in original frame coords.

    Writes a per-frame CSV and a 15-frame contact sheet; returns a summary dict.
    """
    os.makedirs(outdir, exist_ok=True)
    hit = wrong = miss = 0
    rows = []
    for (idx, _), r, p in zip(frames, refs, preds):
        if p is None:
            miss += 1
            status = "miss"
        elif r is not None and np.hypot(p[0] - r[0], p[1] - r[1]) <= HIT_RADIUS_PX:
            hit += 1
            status = "hit"
        else:
            wrong += 1
            status = "wrong"
        rows.append([idx, status, *(p or ("", "", "")), *(r or ("", ""))])
    with open(os.path.join(outdir, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "status", "pred_x", "pred_y", "conf", "ref_x", "ref_y"])
        w.writerows(rows)

    tiles = []
    for j in range(0, len(frames), 8):
        vis = frames[j][1].copy()
        if refs[j]:
            cv2.circle(vis, tuple(map(int, refs[j])), HIT_RADIUS_PX, (0, 255, 0), 1)
        if preds[j]:
            cv2.drawMarker(vis, tuple(map(int, preds[j][:2])), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(vis, f"{preds[j][2]:.2f}", (int(preds[j][0]) + 8, int(preds[j][1]) - 8), 0, 0.5, (0, 0, 255), 1)
        cv2.putText(vis, rows[j][1], (10, 25), 0, 0.8, (0, 0, 255), 2)
        tiles.append(cv2.resize(vis, (320, 240)))
    tiles = tiles[:15] + [np.zeros_like(tiles[0])] * (15 - len(tiles[:15]))
    sheet = np.vstack([np.hstack(tiles[i * 5:(i + 1) * 5]) for i in range(3)])
    cv2.imwrite(os.path.join(outdir, f"{name}_sheet.jpg"), sheet)

    s = {"model": name, "hit": hit, "wrong": wrong, "miss": miss, "n": len(frames)}
    print(f"{name:40s} hit {hit:3d}  wrong {wrong:3d}  miss {miss:3d}  / {len(frames)}", flush=True)
    return s


# --- YOLOE visual prompt helpers ---

def wall_distance(r, arena=ARENA):
    x1, y1, x2, y2 = arena
    return min(r[0] - x1, x2 - r[0], r[1] - y1, y2 - r[1])


def pick_examples(refs, n, arena=ARENA):
    """Frames whose mouse positions are spread out: most central first, then nearest to walls/corners."""
    cand = [j for j, r in enumerate(refs) if r]
    cand.sort(key=lambda j: -wall_distance(refs[j], arena))
    chosen = [cand[0]]
    while len(chosen) < n:  # farthest-point sampling over positions
        chosen.append(max(cand, key=lambda j: min(np.hypot(refs[j][0] - refs[c][0], refs[j][1] - refs[c][1]) for c in chosen)))
    return chosen


def build_reference(frames, refs, examples, arena=ARENA, half=HALF):
    x1, y1, x2, y2 = arena
    w, h = x2 - x1, y2 - y1
    cols = 1 if len(examples) == 1 else 2
    rows = int(np.ceil(len(examples) / cols))
    canvas = np.full((rows * h, cols * w, 3), 127, np.uint8)
    boxes = []
    for k, j in enumerate(examples):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[j][1][y1:y2, x1:x2]
        cx, cy = refs[j][0] - x1 + c * w, refs[j][1] - y1 + r * h
        boxes.append([cx - half, cy - half, cx + half, cy + half])
    return canvas, np.array(boxes, float)


def best_box(res, max_size=90, arena=ARENA):
    best = None
    for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x1, y1, x2, y2 = arena
        small = max(b[2] - b[0], b[3] - b[1]) <= max_size
        if x1 <= cx <= x2 and y1 <= cy <= y2 and small and (best is None or c > best[2]):
            best = (cx, cy, c, b)
    return best
