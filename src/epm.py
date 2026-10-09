"""Elevated plus maze geometry, zones and measures.

The maze is described by 12 image points named as in the DLCAnalyzer EPM network (so the reference data and
our UI use the same layout): closed-arm ends tl tr (top) and bl br (bottom), open-arm ends lt lb (left) and
rt rb (right), centre-square corners ctl ctr cbl cbr. In the ideal template the closed arms are vertical.

The user either drags the 4 arm tips (and types the arm width) -> the 12 points are generated, or drags all
12 points. Pixel -> cm uses a homography fitted from the 12 points to the template, so a tilted camera is
handled like the 4 floor corners of the open field.
"""
import cv2
import numpy as np
import pandas as pd
from matplotlib.path import Path

from src.metrics import moving_flags, runs, step_lengths, max_speed, zone_measures

POINTS = ["tl", "tr", "ctr", "rt", "rb", "cbr", "br", "bl", "cbl", "lb", "lt", "ctl"]  # outline order
ZONES = {  # DLCAnalyzer EPM_zoneinfo.csv
    "closed_top": ["tl", "tr", "ctr", "ctl"],
    "closed_bottom": ["cbr", "br", "bl", "cbl"],
    "open_left": ["lt", "ctl", "cbl", "lb"],
    "open_right": ["rt", "rb", "cbr", "ctr"],
    "centre": ["ctr", "ctl", "cbl", "cbr"],
}
GROUPS = {"open": ["open_left", "open_right"], "closed": ["closed_top", "closed_bottom"], "centre": ["centre"],
          "open_left": ["open_left"], "open_right": ["open_right"],
          "closed_top": ["closed_top"], "closed_bottom": ["closed_bottom"]}
TIPS = ["closed 1", "open 1", "closed 2", "open 2"]  # drag order in the UI: top, right, bottom, left


def template_cm(span_cm=65.5, width_cm=5.5):
    """The 12 points of an ideal '+' in cm, origin at the maze centre, closed arms vertical."""
    a, h = span_cm / 2, width_cm / 2
    return {"tl": (-h, -a), "tr": (h, -a), "ctl": (-h, -h), "ctr": (h, -h), "cbl": (-h, h), "cbr": (h, h),
            "bl": (-h, a), "br": (h, a), "lt": (-a, -h), "lb": (-a, h), "rt": (a, -h), "rb": (a, h)}


def tip_template_cm(span_cm=65.5):
    a = span_cm / 2
    return np.float32([[0, -a], [a, 0], [0, a], [-a, 0]])  # top, right, bottom, left arm-end midpoints


def points_from_tips(tips_px, span_cm=65.5, width_cm=5.5):
    """12 image points from the 4 arm tips (top, right, bottom, left) and the arm width."""
    H = cv2.getPerspectiveTransform(tip_template_cm(span_cm), np.float32(tips_px))
    t = template_cm(span_cm, width_cm)
    pts = cv2.perspectiveTransform(np.float32([t[p] for p in POINTS]).reshape(-1, 1, 2), H).reshape(-1, 2)
    return {p: pts[i].tolist() for i, p in enumerate(POINTS)}


def tips_from_points(points_px):
    """The 4 arm-tip midpoints of a 12-point maze (to switch from 12-corner mode back to tip mode)."""
    mid = lambda a, b: ((np.array(points_px[a]) + np.array(points_px[b])) / 2).tolist()  # noqa: E731
    return [mid("tl", "tr"), mid("rt", "rb"), mid("bl", "br"), mid("lt", "lb")]


def default_tips(width_px, height_px, fraction=0.8):
    cx, cy = width_px / 2, height_px / 2
    a = fraction * min(width_px, height_px) / 2
    return [[cx, cy - a], [cx + a, cy], [cx, cy + a], [cx - a, cy]]


class MazeCalibration:
    """Pixel <-> cm for the maze: homography fitted from the 12 image points to the template (least squares)."""

    def __init__(self, points_px, span_cm=65.5, width_cm=5.5):
        self.points_px = {p: list(map(float, points_px[p])) for p in POINTS}
        self.span_cm, self.width_cm = span_cm, width_cm
        t = template_cm(span_cm, width_cm)
        src = np.float32([self.points_px[p] for p in POINTS])
        dst = np.float32([t[p] for p in POINTS])
        self.H, _ = cv2.findHomography(src, dst, 0)
        self.corners_px = src  # outline, used for drawing and the detector crop

    def to_cm(self, xy_px):
        pts = np.asarray(xy_px, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_px(self, xy_cm):
        pts = np.asarray(xy_cm, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def crop_box(self, frame_shape, pad_fraction=0.1):
        x1, y1 = self.corners_px.min(axis=0)
        x2, y2 = self.corners_px.max(axis=0)
        px, py = (x2 - x1) * pad_fraction, (y2 - y1) * pad_fraction
        h, w = frame_shape[:2]
        return int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))

    def valid_region_px(self, margin_cm=5.0):
        """Where the mouse can be: the '+' widened and lengthened by margin_cm (head dips, walls)."""
        t = template_cm(self.span_cm + 2 * margin_cm, self.width_cm + 2 * margin_cm)
        return Path(self.to_px([t[p] for p in POINTS]))

    def inclusion_px(self, scale=1.8):
        """Maze outline scaled about its centre (DLCAnalyzer drops tracked points outside it)."""
        c = self.corners_px.mean(axis=0)
        return Path(c + scale * (self.corners_px - c))


class MazeZones:
    """EPM zones in cm, from the user's 12 points mapped through the calibration."""

    def __init__(self, calib):
        self.calib = calib
        cm = {p: calib.to_cm([calib.points_px[p]])[0] for p in POINTS}
        self.polygons = {z: np.array([cm[p] for p in pts]) for z, pts in ZONES.items()}
        self.outline = np.array([cm[p] for p in POINTS])
        a = calib.span_cm / 2
        self.extent = (-a, a, a, -a)  # x0, x1, y0 (bottom), y1 (top) for imshow

    def in_zone(self, name, xy_cm):
        xy = np.asarray(xy_cm, float).reshape(-1, 2)
        valid = ~np.isnan(xy).any(axis=1)
        inside = np.any([Path(self.polygons[z]).contains_points(np.nan_to_num(xy)) for z in GROUPS[name]], axis=0)
        return inside & valid

    def drawing(self):
        """Zone polygons (cm) to draw: name -> list of polygons."""
        return {"open": [self.polygons["open_left"].tolist(), self.polygons["open_right"].tolist()],
                "closed": [self.polygons["closed_top"].tolist(), self.polygons["closed_bottom"].tolist()],
                "centre": [self.polygons["centre"].tolist()]}

    def grid_lines_cm(self):
        return []


def epm_measures(xy_cm, t, fps, zones, movement_cutoff=5.0, integration_period=5, immobile_min_s=2.0,
                 max_speed_min_cm=2.0, bin_s=60.0):
    """Returns (summary, zone table, per-bin table, per-frame table), like metrics.open_field."""
    xy_cm = np.asarray(xy_cm, float)
    t = np.asarray(t, float)
    duration = len(xy_cm) / fps
    valid = ~np.isnan(xy_cm).any(axis=1)
    step = step_lengths(xy_cm)
    moving = moving_flags(step, fps, movement_cutoff, integration_period)
    immobile = [(s, e) for s, e in runs(~moving) if (e - s) / fps >= immobile_min_s]

    flags = {name: zones.in_zone(name, xy_cm) for name in GROUPS}
    table = pd.DataFrame.from_dict({name: zone_measures(f, step, t, fps, integration_period)
                                    for name, f in flags.items()}, orient="index")
    table.index.name = "zone"
    open_t, closed_t = table.loc["open", "time_s"], table.loc["closed", "time_s"]
    open_e, closed_e = table.loc["open", "entries"], table.loc["closed", "entries"]
    summary = {
        "test_duration_s": round(duration, 2),
        "frames_tracked_pct": round(100 * float(valid.mean()), 1),
        "open_time_pct": round(100 * open_t / (open_t + closed_t), 1) if open_t + closed_t else None,
        "open_entries_pct": round(100 * open_e / (open_e + closed_e), 1) if open_e + closed_e else None,
        "arm_entries": int(open_e + closed_e),
        "total_distance_cm": round(float(step.sum()), 1),
        "distance_moving_cm": round(float(step[moving].sum()), 1),
        "average_speed_cm_s": round(float(step.sum() / duration), 2),
        "max_speed_cm_s": round(max_speed(xy_cm, t, max_speed_min_cm), 1),
        "time_moving_s": round(float(moving.sum() / fps), 1),
        "time_immobile_s": round(sum(e - s for s, e in immobile) / fps, 1),
        "immobile_episodes": len(immobile),
    }

    b = ((t - t[0]) // bin_s).astype(int)
    keys = np.unique(b)
    per_bin = pd.DataFrame({
        "bin_start_s": [k * bin_s for k in keys],
        "distance_cm": [round(float(step[b == k].sum()), 1) for k in keys],
        "time_moving_s": [round(float(moving[b == k].sum() / fps), 1) for k in keys],
        "open_time_s": [round(float(flags["open"][b == k].sum() / fps), 1) for k in keys],
        "frames_tracked_pct": [round(100 * float(valid[b == k].mean()), 1) for k in keys],
    })
    per_frame = pd.DataFrame({"time_s": t, "x_cm": xy_cm[:, 0], "y_cm": xy_cm[:, 1], "speed_cm_s": step * fps,
                              "moving": moving, **{f"in_{n}": flags[n] for n in ("open", "closed", "centre")}})
    return summary, table, per_bin, per_frame
