"""One analysis run: cached detections + a setup (arena, zones, timing) -> open-field results.

Shared by scripts/analyze_video.py and the UI server, so both always give the same numbers. The detector runs
once per video (track_video); everything here is fast, so the UI can recompute on every setup change.
"""
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from src.calibration import Calibration
from src.metrics import open_field
from src.tracking import auto_end_s, auto_start_s, clean_track
from src.zones import ZoneSpec


@dataclass
class Setup:
    corners_px: list                      # floor corners TL, TR, BR, BL in image pixels
    width_cm: float = 40.0
    height_cm: float = 40.0
    grid_cm: float = 10.0
    center_mode: str = "percent"          # "percent" of the side, or "cm" from the walls
    center_value: float = 50.0
    toggled_cells: list = field(default_factory=list)
    corner_zones: bool = True
    start_s: float | None = None          # None = automatic
    end_s: float | None = None
    corners_source: str = ""              # where the corners came from, shown in the UI

    def calibration(self):
        return Calibration(self.corners_px, self.width_cm, self.height_cm)

    def zones(self, corner_fraction=0.2):
        return ZoneSpec(self.width_cm, self.height_cm, self.grid_cm, self.center_mode, self.center_value,
                        self.toggled_cells, self.corner_zones, corner_fraction)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def analyze(raw, fps, setup, cfg):
    """raw: DataFrame from track_video (frame, time_s, x_px, y_px, conf, foreground_px).
    cfg: the config.yaml dict. Returns a dict with timing, summary, zone table, per-bin and per-frame tables."""
    raw = raw.copy()
    raw.attrs["fps"] = fps
    tc, oc = cfg["tracking"], cfg["open_field"]
    calib = setup.calibration()
    track = clean_track(raw, calib, tc["max_speed_cm_s"], tc["max_gap_s"], tc["smooth_s"])

    auto_start = auto_start_s(raw, tc["hand_foreground_px"])
    auto_end = auto_end_s(raw, tc["hand_foreground_px"])
    start = auto_start if setup.start_s is None else float(setup.start_s)
    end = auto_end if setup.end_s is None else float(setup.end_s)
    win = track[(track.time_s >= start) & (track.time_s <= end)]
    if len(win) < 2:
        raise ValueError(f"analysis window {start:.1f}-{end:.1f} s contains no frames")

    zones = setup.zones(oc["corner_fraction"])
    summary, table, per_bin, per_frame = open_field(
        win[["x_cm", "y_cm"]].to_numpy(), win.time_s.to_numpy(), fps, zones,
        oc["movement_cutoff_cm_s"], oc["integration_period_frames"], oc["immobile_min_s"],
        oc["max_speed_min_cm"], oc["bin_s"])
    per_frame.insert(0, "frame", win.frame.to_numpy())
    per_frame["x_px"], per_frame["y_px"] = win.x_px.to_numpy(), win.y_px.to_numpy()
    summary = {"start_s": round(start, 2), "end_s": round(end, 2), **summary}
    return {"auto_start_s": round(auto_start, 2), "auto_end_s": round(auto_end, 2), "summary": summary,
            "zones": table, "per_bin": per_bin, "per_frame": per_frame, "zone_spec": zones, "calibration": calib}


def default_corners(width_px, height_px, fraction=0.6):
    """A centred square-ish guess used when no saved setup exists; the user drags it onto the floor corners."""
    side = fraction * min(width_px, height_px)
    cx, cy = width_px / 2, height_px / 2
    h = side / 2
    return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]


def nan_safe(v):
    """For JSON output: NaN/inf -> None, numpy scalars -> Python."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v
