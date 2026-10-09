"""Mark the 4 floor corners of the arena and save the calibration.

Interactive (click top-left, top-right, bottom-right, bottom-left floor corners; r = restart, Enter = save):
    .venv\\Scripts\\python scripts\\calibrate.py data\\ysmu\\video.mp4 calib\\ysmu.json --size 40 40
Non-interactive:
    .venv\\Scripts\\python scripts\\calibrate.py data\\ysmu\\video.mp4 calib\\ysmu.json --size 40 40 --corners 161,92 434,89 427,364 141,359

Writes the JSON and a preview image (same name, .jpg) showing the floor outline and a 5 cm grid.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.calibration import Calibration  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--size", nargs=2, type=float, required=True, metavar=("WIDTH_CM", "HEIGHT_CM"),
                help="inner floor size; width is the top edge as seen in the image")
ap.add_argument("--corners", nargs=4, help="x,y pixel pairs: TL TR BR BL")
ap.add_argument("--frame", type=float, default=0.5, help="where in the video to grab the frame (0-1)")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * args.frame))
ok, frame = cap.read()
if not ok:
    sys.exit(f"cannot read {args.video}")

if args.corners:
    pts = [tuple(map(float, c.split(","))) for c in args.corners]
else:
    scale = 2  # show enlarged so corners can be placed precisely
    big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pts = []
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while True:
        vis = big.copy()
        for k, p in enumerate(pts):
            cv2.circle(vis, (int(p[0] * scale), int(p[1] * scale)), 5, (0, 0, 255), 1)
            if k:
                cv2.line(vis, (int(pts[k - 1][0] * scale), int(pts[k - 1][1] * scale)),
                         (int(p[0] * scale), int(p[1] * scale)), (0, 0, 255), 1)
        msg = f"click {names[len(pts)]} floor corner" if len(pts) < 4 else "Enter = save, r = restart"
        cv2.putText(vis, msg, (10, 25), 0, 0.7, (0, 0, 255), 2)
        cv2.imshow("calibrate", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            pts.clear()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            sys.exit("cancelled")
    cv2.destroyAllWindows()

calib = Calibration(pts, *args.size)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
calib.save(args.out)

# preview: floor outline + 5 cm grid projected back into the image
vis = frame.copy()
w, h = args.size
for gx in np.arange(0, w + 0.01, 5):
    a, b = calib.to_px([[gx, 0], [gx, h]]).astype(int)
    cv2.line(vis, tuple(a), tuple(b), (0, 200, 0), 1)
for gy in np.arange(0, h + 0.01, 5):
    a, b = calib.to_px([[0, gy], [w, gy]]).astype(int)
    cv2.line(vis, tuple(a), tuple(b), (0, 200, 0), 1)
cv2.polylines(vis, [calib.corners_px.astype(int)], True, (0, 0, 255), 1)
cv2.imwrite(os.path.splitext(args.out)[0] + ".jpg", vis)
print(f"saved {args.out} (+ preview .jpg)")
