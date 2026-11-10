"""Open-field metrics from a track in floor cm, following the DLCAnalyzer rules (validated against Sturman et al.)
and reported with ANY-maze style names.

DLCAnalyzer rules (R/DLCAnalyzer_Functions_final.R): per-frame step length is the speed; a frame is "moving" when
the step exceeds movement_cutoff / fps, smoothed by a +-integration_period majority vote (avgbool); time in a zone
counts raw frames in the zone; zone transitions are counted on the smoothed in-zone flag.
"""
import numpy as np
import pandas as pd


def avgbool(x, window):
    """DLCAnalyzer avgbool: mean over +-window frames (clipped at the ends) > 0.5."""
    x = np.asarray(x, float)
    c = np.r_[0, np.cumsum(x)]
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - window), np.minimum(len(x), i + window + 1)
    return (c[hi] - c[lo]) / (hi - lo) > 0.5


def step_lengths(xy):
    """Distance moved since the previous frame (first frame 0); steps touching a NaN count as 0."""
    xy = np.asarray(xy, float)
    step = np.r_[0.0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))]
    return np.nan_to_num(step)


def moving_flags(step, fps, movement_cutoff=5.0, integration_period=5):
    return avgbool(step > movement_cutoff / fps, integration_period)


def runs(flag):
    """(start, end) index pairs of consecutive True runs (end exclusive)."""
    f = np.r_[False, np.asarray(flag, bool), False]
    d = np.flatnonzero(np.diff(f.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


def max_speed(xy, t, min_distance_cm=2.0):
    """Fastest speed over a stretch of at least min_distance_cm of path, so jitter does not count."""
    cum = np.cumsum(step_lengths(xy))
    j = np.searchsorted(cum, cum + min_distance_cm)  # first frame at least min_distance further along the path
    ok = j < len(cum)
    if not ok.any():
        return 0.0
    i = np.flatnonzero(ok)
    dt = t[j[ok]] - t[i]
    return float(np.max((cum[j[ok]] - cum[i]) / np.maximum(dt, 1e-9)))


def min_move_distance(xy, min_move):
    """Path length counting a move only once the point is min_move away from the last counted point, so jitter
    while resting adds nothing. Approximates ANY-maze's (unpublished) size-scaled smoothing; fitted on 1 YSMU video."""
    total, anchor = 0.0, None
    for p in np.asarray(xy, float):
        if np.isnan(p).any():
            continue
        if anchor is None:
            anchor = p
            continue
        d = float(np.hypot(*(p - anchor)))
        if d >= min_move:
            total += d
            anchor = p
    return total


def zone_measures(inside, step, t, fps, integration_period):
    """ANY-maze style measures for one zone. Time and distance use raw frames; entries, exits, latency and
    visits use the smoothed flag (as DLCAnalyzer counts transitions)."""
    smooth = avgbool(inside, integration_period)
    visits = runs(smooth)
    entries = len(visits)
    time_in = float(np.sum(inside) / fps)
    return {
        "time_s": round(time_in, 2),
        "time_pct": round(100 * time_in / (len(inside) / fps), 1) if len(inside) else 0.0,
        "entries": entries,
        "exits": sum(1 for _, e in visits if e < len(smooth)),
        "latency_s": round(float(t[visits[0][0]] - t[0]), 2) if visits else None,
        "distance_cm": round(float(np.sum(step[inside])), 1),
        "avg_visit_s": round(time_in / entries, 2) if entries else None,
        "transitions": int(np.sum(smooth[1:] != smooth[:-1])),
    }


def open_field(xy_cm, t, fps, zones, movement_cutoff=5.0, integration_period=5, immobile_min_s=2.0,
               max_speed_min_cm=2.0, bin_s=60.0, anymaze_min_move_cm=4.0):
    """xy_cm: (N, 2) floor positions for the analysis window (NaN where the mouse was not found).
    zones: a src.zones.ZoneSpec. Returns (summary dict, zone table, per-time-bin table, per-frame table)."""
    xy_cm = np.asarray(xy_cm, float)
    t = np.asarray(t, float)
    duration = len(xy_cm) / fps
    valid = ~np.isnan(xy_cm).any(axis=1)
    step = step_lengths(xy_cm)
    moving = moving_flags(step, fps, movement_cutoff, integration_period)
    immobile = [(s, e) for s, e in runs(~moving) if (e - s) / fps >= immobile_min_s]

    cells = zones.grid_cell(xy_cm)
    seen = cells[cells >= 0]

    summary = {
        "test_duration_s": round(duration, 2),
        "frames_tracked_pct": round(100 * float(valid.mean()), 1),
        "total_distance_cm": round(float(step.sum()), 1),
        "distance_moving_cm": round(float(step[moving].sum()), 1),
        "distance_anymaze_cm": round(min_move_distance(xy_cm, anymaze_min_move_cm), 1),
        "average_speed_cm_s": round(float(step.sum() / duration), 2),
        "max_speed_cm_s": round(max_speed(xy_cm, t, max_speed_min_cm), 1),
        "time_moving_s": round(float(moving.sum() / fps), 1),
        "time_immobile_s": round(sum(e - s for s, e in immobile) / fps, 1),
        "immobile_episodes": len(immobile),
        "line_crossings": int(np.sum(seen[1:] != seen[:-1])),
    }

    flags = {name: zones.in_zone(name, xy_cm) for name in zones.masks}
    table = pd.DataFrame.from_dict({name: zone_measures(f, step, t, fps, integration_period)
                                    for name, f in flags.items()}, orient="index")
    table.index.name = "zone"

    b = ((t - t[0]) // bin_s).astype(int)
    keys = np.unique(b)
    per_bin = pd.DataFrame({
        "bin_start_s": [k * bin_s for k in keys],
        "distance_cm": [round(float(step[b == k].sum()), 1) for k in keys],
        "time_moving_s": [round(float(moving[b == k].sum() / fps), 1) for k in keys],
        "centre_time_s": [round(float(flags["centre"][b == k].sum() / fps), 1) for k in keys],
        "frames_tracked_pct": [round(100 * float(valid[b == k].mean()), 1) for k in keys],
    })
    per_frame = pd.DataFrame({"time_s": t, "x_cm": xy_cm[:, 0], "y_cm": xy_cm[:, 1], "speed_cm_s": step * fps,
                              "moving": moving, **{f"in_{name}": f for name, f in flags.items()}})
    return summary, table, per_bin, per_frame
