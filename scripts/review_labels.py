"""Offline review of draft mouse boxes. Accepted boxes are saved as YOLO labels.

    .venv\\Scripts\\python scripts\\review_labels.py data\\labels\\ysmu_v1

Keys:
    Enter / Space   accept the box shown and go to the next frame
    drag (mouse)    draw a new box (head + body; leave the tail out)
    n               no mouse visible (saves an empty label)
    r               reset to the draft box
    a / d           previous / next frame without saving
    q / Esc         quit (progress is saved; rerun to resume at the first unreviewed frame)

A frame counts as reviewed once labels/<name>.txt exists.
"""
import sys
from pathlib import Path

import cv2

SCALE = 2
ZOOM = 4
root = Path(sys.argv[1])
names = sorted(p.stem for p in (root / "images").glob("*.jpg"))
WIN = "review  |  Enter=accept  drag=redraw  n=no mouse  r=reset  a/d=prev/next  q=quit"


def read_box(path, W, H):
    """YOLO txt -> (x1, y1, x2, y2) in pixels, or None."""
    if not path.exists() or not path.read_text().strip():
        return None
    _, cx, cy, w, h = map(float, path.read_text().split()[:5])
    return [int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)]


def write_box(path, box, W, H):
    if box is None:
        path.write_text("")
        return
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, x1), min(W, x2)))
    y1, y2 = sorted((max(0, y1), min(H, y2)))
    path.write_text(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")


state = {"box": None, "drag": None}


def on_mouse(event, x, y, flags, _):
    x, y = x // SCALE, y // SCALE
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drag"] = (x, y)
        state["box"] = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
        x0, y0 = state["drag"]
        state["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
    elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
        state["drag"] = None
        if state["box"][2] - state["box"][0] < 3 or state["box"][3] - state["box"][1] < 3:
            state["box"] = None  # a click, not a drag


def render(img, i, reviewed):
    H, W = img.shape[:2]
    vis = cv2.resize(img, (W * SCALE, H * SCALE), interpolation=cv2.INTER_LINEAR)
    box = state["box"]
    if box:
        x1, y1, x2, y2 = (v * SCALE for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # zoomed inset of the box area, top-right
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        r = 30
        crop = img[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r].copy()
        ox, oy = max(cx - r, 0), max(cy - r, 0)
        cv2.rectangle(crop, (box[0] - ox, box[1] - oy), (box[2] - ox, box[3] - oy), (0, 0, 255), 1)
        crop = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM), interpolation=cv2.INTER_NEAREST)
        vis[0:crop.shape[0], vis.shape[1] - crop.shape[1]:] = crop
    done = sum((root / "labels" / f"{n}.txt").exists() for n in names)
    status = f"{i + 1}/{len(names)}  {names[i]}  {'REVIEWED' if reviewed else 'draft'}  {'NO MOUSE' if box is None else ''}  done {done}"
    cv2.rectangle(vis, (0, vis.shape[0] - 28), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, status, (8, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def load(i):
    img = cv2.imread(str(root / "images" / f"{names[i]}.jpg"))
    H, W = img.shape[:2]
    label = root / "labels" / f"{names[i]}.txt"
    reviewed = label.exists()
    state["box"] = read_box(label if reviewed else root / "drafts" / f"{names[i]}.txt", W, H)
    return img, reviewed


i = next((k for k, n in enumerate(names) if not (root / "labels" / f"{n}.txt").exists()), 0)
cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, on_mouse)
img, reviewed = load(i)
while True:
    cv2.imshow(WIN, render(img, i, reviewed))
    k = cv2.waitKey(20) & 0xFF
    if k == 255:
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        continue
    H, W = img.shape[:2]
    if k in (13, 32):
        write_box(root / "labels" / f"{names[i]}.txt", state["box"], W, H)
        i = min(i + 1, len(names) - 1)
    elif k == ord("n"):
        write_box(root / "labels" / f"{names[i]}.txt", None, W, H)
        i = min(i + 1, len(names) - 1)
    elif k == ord("r"):
        state["box"] = read_box(root / "drafts" / f"{names[i]}.txt", W, H)
        continue
    elif k == ord("a"):
        i = max(i - 1, 0)
    elif k == ord("d"):
        i = min(i + 1, len(names) - 1)
    elif k in (ord("q"), 27):
        break
    else:
        continue
    img, reviewed = load(i)
cv2.destroyAllWindows()
done = sum((root / "labels" / f"{n}.txt").exists() for n in names)
print(f"reviewed {done}/{len(names)}")
