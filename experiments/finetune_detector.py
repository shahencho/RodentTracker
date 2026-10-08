"""Fine-tune a YOLO11n mouse detector from the KUNIN mouse-pose backbone on YOLOE auto-labels.

Split by time: the last 30% of each video is validation (frames are near-duplicates of
their neighbours, so a random split would leak). Frames flagged as a jump are left out.
Then scores the model on the 120 probe frames of each video against the background-
subtraction reference, separately for the held-out part.

Usage: .venv\\Scripts\\python experiments\\finetune_detector.py
"""
import csv
import glob
import os
import shutil
import sys
import time

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from probe_common import ARENA, hand_in_view, median_background, reference_positions, sample_frames, score  # noqa: E402

LABELS = os.path.join("data", "labels", "yoloe")
DATASET = os.path.join("data", "datasets", "ysmu_yoloe")
VAL_FRACTION = 0.3
KUNIN = "models/third_party/kunin/models/kunin-mice-pose.v0.1.5n.pt"
VIDEOS = sorted(glob.glob(os.path.join("data", "ysmu", "*.mp4")))
PAD = 10  # px kept around the arena so a mouse rearing on the wall is not cut off
IMGSZ = 352  # arena crop is ~360 px, so the mouse stays at native resolution (3x faster than 640 on CPU)


def crop_box():
    x1, y1, x2, y2 = ARENA
    return x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD


def crop(frame):
    x1, y1, x2, y2 = crop_box()
    return frame[y1:y2, x1:x2]


def build_dataset():
    shutil.rmtree(DATASET, ignore_errors=True)
    counts = {"train": 0, "val": 0, "hand_skipped": 0}
    for video in VIDEOS:
        name = os.path.splitext(os.path.basename(video))[0]
        total = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
        with open(os.path.join(LABELS, name, "detections.csv")) as fh:
            keep = [int(r["frame"]) for r in csv.DictReader(fh) if r["flag"] in ("", "jump")]  # jumps checked: fast moves
        fills = os.path.join(LABELS, name, "fills.csv")
        if os.path.exists(fills):  # YOLOE misses filled from background subtraction
            with open(fills) as fh:
                keep += [int(r["frame"]) for r in csv.DictReader(fh) if r["source"] == "bgsub"]
        bg = median_background(video)
        for idx in sorted(keep):
            stem = f"{name}_{idx:06d}"
            if hand_in_view(cv2.imread(os.path.join(LABELS, name, "images", stem + ".jpg")), bg):
                counts["hand_skipped"] += 1  # mouse being placed or picked up: not what we track
                continue
            split = "val" if idx >= (1 - VAL_FRACTION) * total else "train"
            img = cv2.imread(os.path.join(LABELS, name, "images", stem + ".jpg"))
            h, w = img.shape[:2]
            _, cx, cy, bw, bh = map(float, open(os.path.join(LABELS, name, "labels", stem + ".txt")).read().split())
            ox, oy, ox2, oy2 = crop_box()
            cw, ch = ox2 - ox, oy2 - oy
            for kind in ("images", "labels"):
                os.makedirs(os.path.join(DATASET, kind, split), exist_ok=True)
            cv2.imwrite(os.path.join(DATASET, "images", split, stem + ".jpg"), crop(img))
            with open(os.path.join(DATASET, "labels", split, stem + ".txt"), "w") as fh:
                fh.write(f"0 {(cx * w - ox) / cw:.6f} {(cy * h - oy) / ch:.6f} {bw * w / cw:.6f} {bh * h / ch:.6f}\n")
            counts[split] += 1
    with open(os.path.join(DATASET, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(DATASET)}\ntrain: images/train\nval: images/val\nnames:\n  0: mouse\n")
    print("dataset:", counts, flush=True)
    return os.path.join(DATASET, "data.yaml")


def detect(model, frame, conf=0.25):
    """Best mouse box on the arena crop, as (cx, cy, conf) in frame coordinates."""
    r = model.predict(crop(frame), imgsz=IMGSZ, conf=conf, verbose=False)[0]
    if not len(r.boxes):
        return None
    j = int(r.boxes.conf.argmax())
    x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
    ox, oy = crop_box()[:2]
    return (x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, float(r.boxes.conf[j])


def evaluate(weights, tag):
    m = YOLO(weights)
    for video in VIDEOS:
        name = os.path.splitext(os.path.basename(video))[0]
        total = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
        frames = sample_frames(video)
        refs = reference_positions(frames)
        t = time.perf_counter()
        preds = [detect(m, f) for _, f in frames]
        ms = (time.perf_counter() - t) / len(frames) * 1000
        out = os.path.join("outputs", "probe", name)
        score(f"{tag}_all", frames, refs, preds, out)
        held = [j for j, (i, _) in enumerate(frames) if i >= (1 - VAL_FRACTION) * total]
        score(f"{tag}_heldout", [frames[j] for j in held], [refs[j] for j in held], [preds[j] for j in held], out)
        print(f"   {name}: {ms:.0f} ms/frame on CPU", flush=True)


if __name__ == "__main__":
    data = build_dataset()
    model = YOLO("yolo11n.yaml").load(KUNIN)  # backbone + neck from the mouse-pose model, new 1-class detect head
    model.train(data=data, epochs=20, imgsz=IMGSZ, batch=16, device="cpu", workers=2, patience=8,
                project=os.path.abspath("outputs/train"), name="kunin_ft", exist_ok=True,
                fliplr=0.5, flipud=0.5, degrees=10, hsv_h=0, hsv_s=0, plots=True)
    evaluate("outputs/train/kunin_ft/weights/best.pt", "kunin_ft")
