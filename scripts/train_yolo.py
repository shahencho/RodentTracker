"""Train a one-class "mouse" YOLO detector on reviewed labels and score it on the held-out test frames.

    .venv\\Scripts\\python scripts\\train_yolo.py data\\labels\\ysmu_v1 [--model yolo11n.pt] [--epochs 100]

Only frames reviewed with review_labels.py (labels/<name>.txt exists) are used. Splits come from manifest.csv.
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--name", default=None)
args = ap.parse_args()

root = Path(args.root).resolve()
splits = {"train": [], "val": [], "test": []}
with open(root / "manifest.csv") as fh:
    for row in csv.DictReader(fh):
        if (root / "labels" / f"{row['name']}.txt").exists():
            splits[row["split"]].append(str(root / "images" / f"{row['name']}.jpg"))
for split, paths in splits.items():
    (root / f"{split}.txt").write_text("\n".join(paths) + "\n")
    print(f"{split}: {len(paths)} reviewed frames")
if not splits["train"] or not splits["val"]:
    raise SystemExit("review some frames first (scripts/review_labels.py)")

data_yaml = root / "data.yaml"
data_yaml.write_text(f"path: {root.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: mouse\n")

name = args.name or f"{root.name}_{Path(args.model).stem}"
model = YOLO(args.model)
model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=16, patience=25, device="cpu",
            workers=2, project=str(Path("outputs/train").resolve()), name=name, exist_ok=True,
            hsv_h=0.0, hsv_s=0.0,  # grayscale video: colour jitter is meaningless
            fliplr=0.5, flipud=0.5, degrees=10)

best = YOLO(Path("outputs/train") / name / "weights" / "best.pt")
if splits["test"]:
    m = best.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device="cpu", project="outputs/train", name=f"{name}_test", exist_ok=True)
    print(f"TEST  precision {m.box.mp:.3f}  recall {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
