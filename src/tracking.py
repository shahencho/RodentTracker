"""Video -> per-frame mouse track (pixels and cm), with gap filling and smoothing.

One animal, so no multi-object tracker is needed: each frame keeps its most confident box, physically
impossible jumps are dropped, short gaps are interpolated and the path is median-smoothed.
"""
import cv2
import numpy as np
import pandas as pd


def median_background(video, n=60):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gray = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * (total - 1) / (n - 1)))
        ok, f = cap.read()
        if ok:
            gray.append(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    return np.median(np.stack(gray), axis=0).astype(np.uint8)


def foreground_px(frame, bg, box):
    """Pixels inside box that differ from the empty-arena background (exposure drift removed)."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    shift = int(np.median(bg)) - int(np.median(g))
    g = cv2.add(g, shift) if shift > 0 else cv2.subtract(g, -shift)
    x1, y1, x2, y2 = box
    d = cv2.absdiff(g, bg)[y1:y2, x1:x2]
    return cv2.countNonZero(cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)[1])


def track_video(video, detector, calib, batch=16, progress=None):
    """Run the detector on every frame. Also records foreground size, used to spot the experimenter's hand."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = median_background(video)
    rows, buf = [], []

    def flush():
        crop = calib.crop_box(buf[0][1].shape)
        for (i, f), det in zip(buf, detector.detect_batch([f for _, f in buf])):
            rows.append((i, i / fps, *(det or (np.nan, np.nan, 0.0)), foreground_px(f, bg, crop)))
        buf.clear()

    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        buf.append((i, f))
        i += 1
        if len(buf) == batch:
            flush()
            if progress:
                progress(i, total)
    if buf:
        flush()
    df = pd.DataFrame(rows, columns=["frame", "time_s", "x_px", "y_px", "conf", "foreground_px"])
    df.attrs["fps"] = fps
    return df


def auto_start_s(df, hand_px=2500, search_s=60, settle_s=1.0, present_s=1.0):
    """Start of the test: after the experimenter's hand has left the arena (searched in the first minute) and
    once the mouse has been detected continuously for present_s."""
    early = df[df.time_s <= search_s]
    hand = early[early.foreground_px > hand_px]
    t0 = 0.0 if hand.empty else float(hand.time_s.max() + settle_s)
    fps = df.attrs.get("fps") or 1 / max(df.time_s.diff().median(), 1e-6)
    need = max(1, int(round(present_s * fps)))
    after = df[df.time_s >= t0]
    seen = after.x_px.notna().to_numpy().astype(int)
    run = np.convolve(seen, np.ones(need, int), "valid")  # detections in each window of `need` frames
    hit = np.flatnonzero(run == need)
    return float(after.time_s.iloc[hit[0]]) if len(hit) else t0


def auto_end_s(df, hand_px=2500, search_s=60, settle_s=1.0):
    """End of the test: just before the experimenter's hand comes back (searched in the last minute),
    otherwise the end of the video."""
    end = float(df.time_s.iloc[-1])
    late = df[df.time_s >= end - search_s]
    hand = late[late.foreground_px > hand_px]
    return end if hand.empty else max(float(df.time_s.iloc[0]), float(hand.time_s.min() - settle_s))


def clean_track(df, calib, max_speed_cm_s=100.0, max_gap_s=0.5, smooth_s=0.0):
    """Add cm coordinates, drop impossible jumps, fill short gaps and optionally median-smooth (off by default:
    the validated DLCAnalyzer comparison used the unsmoothed track). Keeps raw columns for QA."""
    fps = df.attrs["fps"]
    df = df.copy()
    xy = calib.to_cm(df[["x_px", "y_px"]].fillna(0).to_numpy())
    df["x_raw_cm"], df["y_raw_cm"] = xy[:, 0], xy[:, 1]
    df.loc[df.x_px.isna(), ["x_raw_cm", "y_raw_cm"]] = np.nan
    df["detected"] = df.x_px.notna()

    # drop detections that imply a speed no mouse can reach (relative to the last kept point)
    keep = df.detected.to_numpy().copy()
    last = None
    for k in np.flatnonzero(keep):
        p = (df.x_raw_cm.iat[k], df.y_raw_cm.iat[k], df.time_s.iat[k])
        if last and np.hypot(p[0] - last[0], p[1] - last[1]) / max(p[2] - last[2], 1 / fps) > max_speed_cm_s:
            keep[k] = False
            continue
        last = p
    df["jump_rejected"] = df.detected & ~keep
    x = df.x_raw_cm.where(keep)
    y = df.y_raw_cm.where(keep)

    # interpolate gaps up to max_gap_s; longer gaps stay missing
    limit = max(1, int(round(max_gap_s * fps)))
    x = x.interpolate(limit=limit, limit_area="inside")
    y = y.interpolate(limit=limit, limit_area="inside")
    if smooth_s > 0:
        win = max(1, int(round(smooth_s * fps)) | 1)  # odd window
        x = x.rolling(win, center=True, min_periods=1).median().where(x.notna())
        y = y.rolling(win, center=True, min_periods=1).median().where(y.notna())
    df["x_cm"], df["y_cm"] = x, y
    return df
