"""Python port of the DLCAnalyzer open-field pipeline (Sturman et al. 2020), used as the answer key.

Mirrors the README single-file OFT example (R/DLCAnalyzer_Functions_final.R):
    ReadDLCDataFromCSV(fps=25) -> CleanTrackingData(likelihoodcutoff=0.95) -> CutTrackingData(start=100, end=250)
    -> CalibrateTrackingData("area", 42*42, tl,tr,br,bl) -> AddOFTZones(scale_center=0.5)
    -> OFTAnalysis(movement_cutoff=5, integration_period=5, points="bodycentre")
Check: on example/OFT/DLC_Data/OFT_3.csv the README reports raw.distance 4008.34, distance.moving 3078.99,
center.total.time 79.8, center.transitions 58.
"""
import numpy as np
import pandas as pd
from matplotlib.path import Path

CORNERS = ["tl", "tr", "br", "bl"]


def read_dlc(path):
    df = pd.read_csv(path, header=[1, 2], index_col=0)  # rows: bodyparts, coords
    return {bp: df[bp].reset_index(drop=True) for bp in df.columns.get_level_values(0).unique()}


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def avgbool(x, window):
    """R avgbool: mean over +-window entries (clipped at the ends) > 0.5."""
    x = np.asarray(x, float)
    c = np.r_[0, np.cumsum(x)]
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - window), np.minimum(len(x), i + window + 1)
    return (c[hi] - c[lo]) / (hi - lo) > 0.5


def corner_medians(data):
    """Median of each arena corner over all frames, before cleaning (as ReadDLCDataFromCSV does)."""
    return np.array([[data[c].x.median(), data[c].y.median()] for c in CORNERS])


def oft_metrics(xy_px, corners_px, fps=25, arena_cm=42.0, start=100, end=250, movement_cutoff=5,
                integration_period=5, scale_center=0.5):
    """DLCAnalyzer OFT metrics for one point track (already cleaned/interpolated), in pixels."""
    xy_px = np.asarray(xy_px, float)
    keep = np.arange(len(xy_px))[start:]
    keep = keep[:len(keep) - (end + 1)]  # R: keep[-((length(keep) - end):length(keep))] drops end+1 frames
    xy = xy_px[keep]

    px_to_cm = np.sqrt(arena_cm * arena_cm / polygon_area(corners_px))
    xy = xy * px_to_cm
    corners = corners_px * px_to_cm
    centre = corners.mean(axis=0)
    center_poly = Path(centre + scale_center * (corners - centre))

    speed = np.r_[0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))]
    moving = avgbool(speed > movement_cutoff / fps, integration_period)
    in_center = center_poly.contains_points(xy)
    smoothed = avgbool(in_center, integration_period)
    return {
        "raw_distance_cm": float(np.nansum(speed)),
        "distance_moving_cm": float(np.nansum(speed[moving])),
        "time_moving_s": float(moving.sum() / fps),
        "total_time_s": float(len(xy) / fps),
        "center_time_s": float(in_center.sum() / fps),
        "center_transitions": int(np.sum(smoothed[1:] != smoothed[:-1])),
    }


def clean_point(df, likelihood_cutoff=0.95):
    """CleanTrackingData: low-likelihood points -> NA -> linear interpolation (ends held constant)."""
    x = df.x.where(df.likelihood >= likelihood_cutoff)
    y = df.y.where(df.likelihood >= likelihood_cutoff)
    x = x.interpolate(limit_direction="both")
    y = y.interpolate(limit_direction="both")
    return np.c_[x.to_numpy(), y.to_numpy()]


def reference_from_dlc(path, **kw):
    data = read_dlc(path)
    return oft_metrics(clean_point(data["bodycentre"]), corner_medians(data), **kw)


if __name__ == "__main__":
    r = reference_from_dlc("data/dlcanalyzer/example/OFT/DLC_Data/OFT_3.csv")
    expected = {"raw_distance_cm": 4008.339382, "distance_moving_cm": 3078.993566, "center_time_s": 79.8,
                "center_transitions": 58, "time_moving_s": 252.32, "total_time_s": 596.08}
    for k, v in expected.items():
        print(f"{k:20s} ours {r[k]:10.3f}   README {v:10.3f}")
