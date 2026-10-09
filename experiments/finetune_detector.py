"""Fine-tune a YOLO11n mouse detector from the KUNIN mouse-pose backbone on auto-labels (YOLOE + bg-sub fills).

Each source video has its own calibration; training images are cropped exactly like the tracker crops at
run time (Calibration.crop_box), so the mouse appears at the same scale in training and use.
Split by time inside each video: the last 30% is validation (neighbouring frames are near-duplicates, so a
random split would leak). Sturman videos not listed here are never seen and are the test set.

Usage: .venv\\Scripts\\python experiments\\finetune_detector.py [--name mouse_det_v2]
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe_common import hand_in_view, median_background  # noqa: E402
from src.calibration import Calibration  # noqa: E402

LABELS = os.path.join("data", "labels", "yoloe")
VAL_FRACTION = 0.3
KUNIN = "models/third_party/kunin/models/kunin-mice-pose.v0.1.5n.pt"
IMGSZ = 352
# (video, calibration, check for the experimenter's hand)
SOURCES = [
    ("data/ysmu/cntrol60_cage22_an1_OF1.mp4", "calib/ysmu_cage22.json", True),
    ("data/ysmu/cntrol60_cage22_an2_OF1.mp4", "calib/ysmu_cage22.json", True),
] + [(f"data/sturman/oft/{v}.mp4", f"calib/sturman/{v}.json", False)
     for v in ["OFT_5", "OFT_44", "OFT_6", "OFT_12", "OFT_23", "OFT_16"]]


def label_frames(name):
    with open(os.path.join(LABELS, name, "detections.csv")) as fh:
        keep = [int(r["frame"]) for r in csv.DictReader(fh) if r["flag"] in ("", "jump")]
    fills = os.path.join(LABELS, name, "fills.csv")
    if os.path.exists(fills):  # YOLOE misses filled from background subtraction
        with open(fills) as fh:
            keep += [int(r["frame"]) for r in csv.DictReader(fh) if r["source"] == "bgsub"]
    return sorted(keep)


def build_dataset(dataset):
    shutil.rmtree(dataset, ignore_errors=True)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            os.makedirs(os.path.join(dataset, kind, split))
    counts = {}
    for video, calib_path, check_hand in SOURCES:
        name = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        calib = Calibration.load(calib_path)
        ox, oy, ox2, oy2 = calib.crop_box(shape)
        cw, ch = ox2 - ox, oy2 - oy
        bg = median_background(video) if check_hand else None
        c = counts.setdefault(name, {"train": 0, "val": 0, "skipped": 0})
        for idx in label_frames(name):
            stem = f"{name}_{idx:06d}"
            img = cv2.imread(os.path.join(LABELS, name, "images", stem + ".jpg"))
            h, w = img.shape[:2]
            _, cx, cy, bw, bh = map(float, open(os.path.join(LABELS, name, "labels", stem + ".txt")).read().split())
            px, py = cx * w - ox, cy * h - oy
            if check_hand and hand_in_view(img, bg) or not (0 <= px < cw and 0 <= py < ch):
                c["skipped"] += 1  # mouse being placed / picked up, or outside the crop
                continue
            split = "val" if idx >= (1 - VAL_FRACTION) * total else "train"
            cv2.imwrite(os.path.join(dataset, "images", split, stem + ".jpg"), img[oy:oy2, ox:ox2])
            with open(os.path.join(dataset, "labels", split, stem + ".txt"), "w") as fh:
                fh.write(f"0 {px / cw:.6f} {py / ch:.6f} {bw * w / cw:.6f} {bh * h / ch:.6f}\n")
            c[split] += 1
    with open(os.path.join(dataset, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(dataset)}\ntrain: images/train\nval: images/val\nnames:\n  0: mouse\n")
    for k, v in counts.items():
        print(f"  {k:28s} {v}")
    print("  total", {s: sum(v[s] for v in counts.values()) for s in ("train", "val", "skipped")}, flush=True)
    return os.path.join(dataset, "data.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mouse_det_v2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--init", help="start from this detector instead of the KUNIN pose backbone")
    args = ap.parse_args()
    data = build_dataset(os.path.join("data", "datasets", args.name))
    if args.init:
        model = YOLO(args.init)
    else:
        model = YOLO("yolo11n.yaml").load(KUNIN)  # backbone + neck from the mouse-pose model, new 1-class detect head
    model.train(data=data, epochs=args.epochs, imgsz=IMGSZ, batch=16, device="cpu", workers=2, patience=5,
                project=os.path.abspath("outputs/train"), name=args.name, exist_ok=True,
                fliplr=0.5, flipud=0.5, degrees=10, hsv_h=0, hsv_s=0, plots=True)
    shutil.copy(os.path.join("outputs", "train", args.name, "weights", "best.pt"), os.path.join("models", args.name + ".pt"))
    print(f"saved models/{args.name}.pt")
