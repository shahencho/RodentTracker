"""Open-field metrics from a cleaned track (cm coordinates, origin at the top-left floor corner)."""
import numpy as np
import pandas as pd


def center_zone(width_cm, height_cm, center_fraction):
    """Central square/rectangle whose sides are center_fraction of the arena sides: (x1, y1, x2, y2) in cm."""
    mx, my = width_cm * (1 - center_fraction) / 2, height_cm * (1 - center_fraction) / 2
    return mx, my, width_cm - mx, height_cm - my


def zone_bouts(inside, min_frames):
    """Debounced in-zone flag: a run in the other zone only switches the state if it lasts min_frames."""
    inside = np.asarray(inside, bool)
    out = np.zeros(len(inside), bool)
    if not len(inside):
        return out
    cut = np.flatnonzero(np.diff(inside.astype(np.int8))) + 1
    state = inside[0]
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(inside)]):
        if inside[s] != state and e - s >= min_frames:
            state = inside[s]
        out[s:e] = state
    return out


def open_field(df, width_cm, height_cm, center_fraction=0.5, move_threshold_cm_s=2.0, min_zone_s=0.2,
               bin_s=60.0):
    """Returns (summary dict, per-time-bin DataFrame, in-center flag per row). df: analysis window only."""
    fps = df.attrs["fps"]
    valid = df.x_cm.notna().to_numpy()
    x, y, t = df.x_cm.to_numpy(), df.y_cm.to_numpy(), df.time_s.to_numpy()

    step = np.hypot(np.diff(x), np.diff(y))
    step = np.where(valid[1:] & valid[:-1], step, 0.0)
    speed = np.r_[0.0, step * fps]
    moving = speed > move_threshold_cm_s
    step_counted = np.where(moving[1:], step, 0.0)  # ignore sub-threshold jitter while still

    cx1, cy1, cx2, cy2 = center_zone(width_cm, height_cm, center_fraction)
    in_center_raw = valid & (x >= cx1) & (x <= cx2) & (y >= cy1) & (y <= cy2)
    in_center = zone_bouts(in_center_raw, max(1, int(round(min_zone_s * fps)))) & valid
    entries = int(np.sum(in_center[1:] & ~in_center[:-1]) + (1 if len(in_center) and in_center[0] else 0))
    first = np.flatnonzero(in_center)

    duration = len(df) / fps
    summary = {
        "start_s": round(float(t[0]), 2),
        "duration_s": round(duration, 2),
        "tracked_pct": round(100 * valid.mean(), 1),
        "detected_pct": round(100 * df.detected.mean(), 1),
        "distance_cm": round(float(step_counted.sum()), 1),
        "mean_speed_cm_s": round(float(step_counted.sum() / duration), 2),
        "moving_time_s": round(float(moving.sum() / fps), 1),
        "center_time_s": round(float(in_center.sum() / fps), 1),
        "center_time_pct": round(100 * float(in_center.sum()) / max(1, valid.sum()), 1),
        "center_entries": entries,
        "center_latency_s": round(float(t[first[0]] - t[0]), 2) if len(first) else None,
        "periphery_time_s": round(float((valid & ~in_center).sum() / fps), 1),
    }

    rel = t - t[0]
    bins = (rel // bin_s).astype(int)
    per_bin = pd.DataFrame({
        "bin_start_s": [b * bin_s for b in np.unique(bins)],
        "distance_cm": [round(float(step_counted[(bins[1:] == b)].sum()), 1) for b in np.unique(bins)],
        "center_time_s": [round(float(in_center[bins == b].sum() / fps), 1) for b in np.unique(bins)],
        "tracked_pct": [round(100 * float(valid[bins == b].mean()), 1) for b in np.unique(bins)],
    })
    df_zone = pd.Series(in_center, index=df.index, name="in_center")
    return summary, per_bin, df_zone
