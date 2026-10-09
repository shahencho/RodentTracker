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


def foreground_mask(frame, bg, box):
    """Pixels inside box that differ from the empty-scene background (exposure drift removed)."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    shift = int(np.median(bg)) - int(np.median(g))
    g = cv2.add(g, shift) if shift > 0 else cv2.subtract(g, -shift)
    x1, y1, x2, y2 = box
    return cv2.absdiff(g, bg)[y1:y2, x1:x2] > 18


class BoxChooser:
    """Picks the mouse among the detector's boxes. Several dark objects can look like a mouse (a cable, a dark
    patch at an arm end), so: a box counts only inside the apparatus region; it scores higher the more it differs
    from the empty-scene background (the mouse moves, fixtures do not); and a box farther from the last position
    than a mouse can run gets a penalty. The penalty is soft, so a wrong pick can never lock in: a moving mouse
    (high background difference) still outscores a static fixture that happens to be "near"."""

    def __init__(self, calib, fps, max_speed_cm_s=100.0, slack_cm=5.0, memory_s=1.0, fg_weight=0.5,
                 far_penalty=0.3):
        self.calib, self.fps = calib, fps
        self.region = calib.valid_region_px() if hasattr(calib, "valid_region_px") else None
        self.max_speed, self.slack, self.memory = max_speed_cm_s, slack_cm, memory_s
        self.fg_weight, self.far_penalty = fg_weight, far_penalty
        self.last = None  # (x_cm, y_cm, t)

    def choose(self, cands, t):
        """cands: [(cx, cy, conf, fg_fraction)] (see with_foreground); returns (cx, cy, conf) or None."""
        if self.region is not None:
            cands = [c for c in cands if self.region.contains_point(c[:2])]
        if not cands:
            return None
        pos = self.calib.to_cm([[c[0], c[1]] for c in cands])
        recent = self.last is not None and t - self.last[2] <= self.memory
        reach = self.max_speed * (t - self.last[2]) + self.slack if recent else None
        best = None
        for (cx, cy, conf, fgf), p in zip(cands, pos):
            score = conf + self.fg_weight * fgf
            if recent and np.hypot(p[0] - self.last[0], p[1] - self.last[1]) > reach:
                score -= self.far_penalty
            if best is None or score > best[0]:
                best = (score, cx, cy, conf, p)
        self.last = (best[4][0], best[4][1], t)
        return best[1], best[2], best[3]


def with_foreground(cands, fg, crop):
    """Add to each box the fraction of its pixels that differ from the empty-scene background."""
    ox, oy = crop[:2]
    out = []
    for cx, cy, conf, (x1, y1, x2, y2) in cands:
        m = fg[max(0, int(y1) - oy):max(0, int(y2) - oy), max(0, int(x1) - ox):max(0, int(x2) - ox)]
        out.append((cx, cy, conf, float(m.mean()) if m.size else 0.0))
    return out


def track_video(video, detector, calib, batch=16, progress=None, candidates_csv=None):
    """Run the detector on every frame and pick the mouse among its boxes (BoxChooser). Also records the
    foreground size in the apparatus crop, used to spot the experimenter's hand. With candidates_csv, every box
    is also saved (frame, cx, cy, conf, fg) so the choice rules can be replayed without re-running the detector."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = median_background(video)
    chooser = BoxChooser(calib, fps)
    rows, buf, all_cands = [], [], []

    def flush():
        crop = calib.crop_box(buf[0][1].shape)
        for (i, f), cands in zip(buf, detector.candidates_batch([f for _, f in buf])):
            fg = foreground_mask(f, bg, crop)
            cands = with_foreground(cands, fg, crop)
            all_cands.extend((i, *c) for c in cands)
            det = chooser.choose(cands, i / fps)
            rows.append((i, i / fps, *(det or (np.nan, np.nan, 0.0)), int(fg.sum())))
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
    if candidates_csv:
        pd.DataFrame(all_cands, columns=["frame", "cx", "cy", "conf", "fg"]).round(4).to_csv(candidates_csv, index=False)
    return df


def replay(candidates, frames, fps, calib, **chooser_kw):
    """Re-run the box choice on saved candidates (from track_video(candidates_csv=...)); returns x_px, y_px, conf."""
    chooser = BoxChooser(calib, fps, **chooser_kw)
    groups = {f: g[["cx", "cy", "conf", "fg"]].to_numpy() for f, g in candidates.groupby("frame")}
    out = np.full((len(frames), 3), np.nan)
    for k, f in enumerate(frames):
        det = chooser.choose([tuple(r) for r in groups.get(f, [])], f / fps)
        if det:
            out[k] = det
    return out


def hand_threshold(df, hand_px=2500, ratio=5.0):
    """Foreground above this means the experimenter's hand (or the lights) is changing the scene: at least
    hand_px, and at least `ratio` x the video's typical foreground (a mouse alone in a big crop is above 2500)."""
    return max(hand_px, ratio * float(df.foreground_px.median()))


def auto_start_s(df, hand_px=2500, search_s=60, settle_s=1.0, present_s=1.0):
    """Start of the test: after the experimenter's hand has left the arena (searched in the first minute) and
    once the mouse has been detected continuously for present_s."""
    early = df[df.time_s <= search_s]
    hand = early[early.foreground_px > hand_threshold(df, hand_px)]
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
    hand = late[late.foreground_px > hand_threshold(df, hand_px)]
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
