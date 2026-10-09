"""Python port of the DLCAnalyzer open-field pipeline (Sturman et al. 2020), used as the answer key.

Mirrors the README single-file OFT example (R/DLCAnalyzer_Functions_final.R):
    ReadDLCDataFromCSV(fps=25) -> CleanTrackingData(likelihoodcutoff=0.95) -> CutTrackingData(start=100, end=250)
    -> CalibrateTrackingData("area", 42*42, tl,tr,br,bl) -> AddOFTZones(scale_center=0.5)
    -> OFTAnalysis(movement_cutoff=5, integration_period=5, points="bodycentre")
Check: on example/OFT/DLC_Data/OFT_3.csv the README reports raw.distance 4008.34, distance.moving 3078.99,
center.total.time 79.8, center.transitions 58.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.path import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import avgbool  # noqa: E402

CORNERS = ["tl", "tr", "br", "bl"]


def read_dlc(path):
    df = pd.read_csv(path, header=[1, 2], index_col=0)  # rows: bodyparts, coords
    return {bp: df[bp].reset_index(drop=True) for bp in df.columns.get_level_values(0).unique()}


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


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


# --- Elevated plus maze (README "EPM analysis of a single file") ---

EPM_POINTS = ["tl", "tr", "ctr", "rt", "rb", "cbr", "br", "bl", "cbl", "lb", "lt", "ctl"]  # "arena" outline order
EPM_ZONES = {  # EPM_zoneinfo.csv: points spanning each zone, in polygon order
    "closed.top": ["tl", "tr", "ctr", "ctl"],
    "closed.bottom": ["cbr", "br", "bl", "cbl"],
    "open.left": ["lt", "ctl", "cbl", "lb"],
    "open.right": ["rt", "rb", "cbr", "ctr"],
    "center": ["ctr", "ctl", "cbl", "cbr"],
}
EPM_REPORT_ZONES = {"center": ["center"], "open": ["open.left", "open.right"], "closed": ["closed.top", "closed.bottom"],
                    "closed.top": ["closed.top"], "closed.bottom": ["closed.bottom"],
                    "open.left": ["open.left"], "open.right": ["open.right"]}


def epm_medians(data):
    """Median of each maze point over all frames (ReadDLCDataFromCSV median.data)."""
    return {p: np.array([data[p].x.median(), data[p].y.median()]) for p in EPM_POINTS}


def epm_reference(xy_px, likelihood, medians, fps=25, arm_span_cm=65.5, start=100, end=250, movement_cutoff=5,
                  integration_period=5, likelihood_cutoff=0.95, inclusion_scale=1.8):
    """DLCAnalyzer EPM pipeline for one point track: Cut -> Calibrate (tl-br = 65.5 cm) -> AddZones -> Clean
    (likelihood + outside the maze outline scaled 1.8x -> interpolate) -> EPMAnalysis zone reports."""
    xy_px = np.asarray(xy_px, float)
    likelihood = np.asarray(likelihood, float)
    keep = np.arange(len(xy_px))[start:]
    keep = keep[:len(keep) - (end + 1)]
    scale = arm_span_cm / np.hypot(*(medians["tl"] - medians["br"]))
    xy = xy_px[keep] * scale
    pts = {p: v * scale for p, v in medians.items()}
    arena = np.array([pts[p] for p in EPM_POINTS])
    centre = arena.mean(axis=0)
    inclusion = Path(centre + inclusion_scale * (arena - centre))
    bad = (likelihood[keep] < likelihood_cutoff) | np.isnan(xy).any(axis=1)
    bad |= ~inclusion.contains_points(np.nan_to_num(xy), radius=1e-9)
    x = pd.Series(np.where(bad, np.nan, xy[:, 0])).interpolate(limit_direction="both")
    y = pd.Series(np.where(bad, np.nan, xy[:, 1])).interpolate(limit_direction="both")
    zones = {name: np.array([pts[p] for p in names]) for name, names in EPM_ZONES.items()}
    return epm_metrics(np.c_[x, y], zones, fps, movement_cutoff, integration_period)


def epm_metrics(xy_cm, zones, fps=25, movement_cutoff=5, integration_period=5):
    """EPMAnalysis report for one point (cm): distance, movement and ZoneReport for each zone group."""
    xy = np.asarray(xy_cm, float)
    speed = np.r_[0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))]
    moving = avgbool(speed > movement_cutoff / fps, integration_period)
    out = {"raw_distance_cm": float(np.nansum(speed)), "distance_moving_cm": float(np.nansum(speed[moving])),
           "time_moving_s": float(moving.sum() / fps), "total_time_s": float(len(xy) / fps)}
    inside = {name: Path(poly).contains_points(xy) for name, poly in zones.items()}
    for report, members in EPM_REPORT_ZONES.items():
        flag = np.any([inside[m] for m in members], axis=0)
        smoothed = avgbool(flag, integration_period)
        out[f"{report}_time_s"] = float(flag.sum() / fps)
        out[f"{report}_transitions"] = int(np.sum(smoothed[1:] != smoothed[:-1]))
        out[f"{report}_distance_moving_cm"] = float(np.nansum(speed[flag & moving]))
    return out


def epm_reference_from_dlc(path, **kw):
    data = read_dlc(path)
    bc = data["bodycentre"]
    return epm_reference(bc[["x", "y"]].to_numpy(), bc.likelihood.to_numpy(), epm_medians(data), **kw)


if __name__ == "__main__":
    r = reference_from_dlc("data/dlcanalyzer/example/OFT/DLC_Data/OFT_3.csv")
    expected = {"raw_distance_cm": 4008.339382, "distance_moving_cm": 3078.993566, "center_time_s": 79.8,
                "center_transitions": 58, "time_moving_s": 252.32, "total_time_s": 596.08}
    for k, v in expected.items():
        print(f"{k:20s} ours {r[k]:10.3f}   README {v:10.3f}")
    r = epm_reference_from_dlc("data/dlcanalyzer/example/EPM/DLC_Data/EPM_1.csv")
    print("EPM_1 (README: nose dips are not ported)")
    for k, v in {"raw_distance_cm": 1768.007995, "distance_moving_cm": 898.767567, "time_moving_s": 97.88}.items():
        print(f"{k:20s} ours {r[k]:10.3f}   README {v:10.3f}")
    print({k: round(v, 2) for k, v in r.items() if "time_s" in k})
