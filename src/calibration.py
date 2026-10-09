"""Arena calibration: 4 clicked floor corners + real size -> pixel-to-cm homography.

The camera is often tilted, so a single px-per-cm scale is wrong; a perspective transform of the
floor plane is used instead. Corners are the floor corners (where the walls meet the floor), in
order top-left, top-right, bottom-right, bottom-left as seen in the image.
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Calibration:
    corners_px: np.ndarray  # (4, 2) TL, TR, BR, BL floor corners in image pixels
    width_cm: float
    height_cm: float

    def __post_init__(self):
        self.corners_px = np.asarray(self.corners_px, np.float32).reshape(4, 2)
        dst = np.float32([[0, 0], [self.width_cm, 0], [self.width_cm, self.height_cm], [0, self.height_cm]])
        self.H = cv2.getPerspectiveTransform(self.corners_px, dst)

    def to_cm(self, xy_px):
        """(N, 2) pixel points -> (N, 2) cm on the floor plane, origin at the top-left floor corner."""
        pts = np.asarray(xy_px, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_px(self, xy_cm):
        pts = np.asarray(xy_cm, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def crop_box(self, frame_shape, pad_fraction=0.15):
        """Detector crop: the floor's bounding box plus margin for walls and a rearing mouse."""
        x1, y1 = self.corners_px.min(axis=0)
        x2, y2 = self.corners_px.max(axis=0)
        px, py = (x2 - x1) * pad_fraction, (y2 - y1) * pad_fraction
        h, w = frame_shape[:2]
        return int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"corners_px": self.corners_px.round(1).tolist(), "width_cm": self.width_cm,
                       "height_cm": self.height_cm}, fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["corners_px"], d["width_cm"], d["height_cm"])
