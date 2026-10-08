"""Score a SuperAnimal probe run: per frame, did it put a confident mouse inside the arena? Writes a contact sheet.

    .venv-dlc\\Scripts\\python scripts\\probe_superanimal_eval.py <clip.mp4> <result.h5>
"""
import sys

import cv2
import numpy as np
import pandas as pd

clip, h5 = sys.argv[1], sys.argv[2]
df = pd.read_hdf(h5)
bodyparts = df.columns.get_level_values("bodyparts").unique()
xy = df.xs("x", level="coords", axis=1).to_numpy(), df.xs("y", level="coords", axis=1).to_numpy()
lik = df.xs("likelihood", level="coords", axis=1).to_numpy()

# arena interior in the YSMU videos (same box used for the YOLO probe)
def inside(cx, cy):
    return 115 < cx < 445 and 70 < cy < 415

cap = cv2.VideoCapture(clip)
hits, tiles = 0, []
for i in range(len(df)):
    ok, f = cap.read()
    if not ok:
        break
    good = lik[i] > 0.5
    found = False
    if good.sum() >= 5:
        cx, cy = np.nanmedian(xy[0][i][good]), np.nanmedian(xy[1][i][good])
        found = inside(cx, cy)
    hits += found
    if i % 8 == 0:
        for x, y, p in zip(xy[0][i], xy[1][i], lik[i]):
            if p > 0.5 and not np.isnan(x):
                cv2.circle(f, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(f, "found" if found else "MISS", (10, 25), 0, 0.8, (0, 0, 255), 2)
        tiles.append(cv2.resize(f, (320, 240)))
print(f"{h5}: {len(bodyparts)} keypoints; confident mouse inside arena in {hits}/{len(df)} frames")
print("median keypoint likelihood:", float(np.nanmedian(lik)))
tiles = tiles[:15] + [np.zeros_like(tiles[0])] * (15 - len(tiles[:15]))
cv2.imwrite(h5.rsplit(".", 1)[0] + "_grid.jpg", np.vstack([np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(3)]))
