"""Demo server: Setup / Results / Validation for one mouse in the open field or the plus maze.

    .venv\\Scripts\\python -m uvicorn app.server:app --port 8765     (or run_ui.bat)

The detector runs once per video (POST /api/run) and its output is cached; every other call recomputes from the
cache, so changing zones or timing updates the results instantly. One detector run at a time (the online demo
shares a small server).

Login: with DEMO_PASSWORD set (environment or .env in the repo root) every page needs it; without it (this PC)
there is no login. Online deployment: deploy/README.md.
"""
import base64
import glob
import hashlib
import hmac
import json
import os
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src import epm, report  # noqa: E402
from src.analysis import Setup, analyze, default_corners, nan_safe  # noqa: E402
from src.tracking import track_video  # noqa: E402
# src.detector (ultralytics + torch, ~400 MB) is imported only when a detector run starts

CFG = yaml.safe_load(open("config.yaml"))
UI_DIR = os.path.join("outputs", "ui")
SKIP_DIRS = (os.path.join("data", "labels"), os.path.join("data", "datasets"))
RERUN_SHIFT = 0.05  # offer a new detector run when an edge of its crop moved more than this fraction of the crop
app = FastAPI(title="RodentTracker demo")
_jobs = {}  # video -> {"state", "done", "total", "error"}
_lock = threading.Lock()


# ---------- login (only when DEMO_PASSWORD is set) ----------

def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
USER, PASSWORD = os.getenv("DEMO_USER", "demo"), os.getenv("DEMO_PASSWORD", "")
COOKIE = "rodent_session"
# stateless session: survives restarts; a new password logs everyone out
TOKEN = hmac.new(PASSWORD.encode(), USER.encode(), hashlib.sha256).hexdigest()
OPEN_PATHS = ("/login", "/static/style.css")


@app.middleware("http")
async def require_login(request: Request, call_next):
    if (not PASSWORD or request.url.path in OPEN_PATHS
            or hmac.compare_digest(request.cookies.get(COOKIE, ""), TOKEN)):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "login required"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(ROOT, "app", "static", "login.html"))


@app.post("/login")
def login(request: Request, user: str = Form(""), password: str = Form("")):
    if not (hmac.compare_digest(user, USER) and hmac.compare_digest(password, PASSWORD)):
        time.sleep(1)  # slows down guessing
        return RedirectResponse("/login?failed=1", status_code=303)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE, TOKEN, max_age=30 * 24 * 3600, httponly=True, samesite="lax",
                    secure=request.headers.get("x-forwarded-proto") == "https")
    return resp


# ---------- videos and per-video state ----------

def list_videos():
    vids = []
    for p in sorted(glob.glob(os.path.join("data", "**", "*.mp4"), recursive=True)):
        if not p.startswith(SKIP_DIRS):
            vids.append(p.replace("\\", "/"))
    return vids


def check_video(video):
    if video not in list_videos():
        raise HTTPException(404, f"unknown video {video}")
    return video


def state_dir(video):
    stem = os.path.splitext(os.path.basename(video))[0]
    d = os.path.join(UI_DIR, f"{stem}_{hashlib.md5(video.encode()).hexdigest()[:6]}")
    os.makedirs(d, exist_ok=True)
    return d


def video_info(video):
    cap = cv2.VideoCapture(video)
    return {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}


def raw_path(video):
    return os.path.join(state_dir(video), "raw_track.csv")


def detected_setup_path(video):
    """The setup the cached detections were made with."""
    return os.path.join(state_dir(video), "detected_setup.json")


def needs_rerun(video, setup):
    """True when the cached detections no longer fit the setup: another test, or the arena / maze outline moved
    so far that the detector's crop changed noticeably. Zones, grid and timing never need a new run."""
    if not os.path.exists(raw_path(video)):
        return False
    p = detected_setup_path(video)
    if not os.path.exists(p):  # detections from before this file existed: the setup saved then is their setup
        load_setup(video).save(p)
    before = Setup.load(p)
    if before.test != setup.test:
        return True
    info = video_info(video)
    shape = (info["height"], info["width"])
    try:
        a = np.array(before.calibration().crop_box(shape), float)
        b = np.array(setup.calibration().crop_box(shape), float)
    except Exception:  # an unfinished outline: the analysis reports it
        return False
    size = np.array([a[2] - a[0], a[3] - a[1]] * 2)
    return bool((np.abs(a - b) > RERUN_SHIFT * size).any())


def load_setup(video):
    s = _load_setup(video)
    if len(s.maze_tips_px) != 4:  # setups saved before the plus maze existed
        info = video_info(video)
        s.maze_tips_px = epm.default_tips(info["width"], info["height"])
    return s


def _load_setup(video):
    """This video's saved setup; else the most recently saved setup from the same folder (same camera);
    else default corners. corners_source says which, for the notice in the UI."""
    own = os.path.join(state_dir(video), "setup.json")
    if os.path.exists(own):
        s = Setup.load(own)
        s.corners_source = "saved setup of this video"
        return s
    folder = os.path.dirname(video)
    best = None
    for other in list_videos():
        p = os.path.join(UI_DIR, os.path.basename(state_dir(other)), "setup.json")
        if other != video and os.path.dirname(other) == folder and os.path.exists(p):
            if best is None or os.path.getmtime(p) > os.path.getmtime(best[1]):
                best = (other, p)
    d = CFG["defaults"]
    if best:
        s = Setup.load(best[1])
        s.start_s = s.end_s = None  # timing is per video
        s.corners_source = f"copied from the saved setup of {os.path.basename(best[0])} (same folder, same camera)"
        return s
    info = video_info(video)
    return Setup(default_corners(info["width"], info["height"]), d["width_cm"], d["height_cm"], d["grid_cm"],
                 d["center_mode"], d["center_value"], corners_source="none",
                 maze_tips_px=epm.default_tips(info["width"], info["height"]))


def setup_from_body(body):
    allowed = Setup.__dataclass_fields__
    return Setup(**{k: v for k, v in body.items() if k in allowed})


def frame_at(video, t=None, index=None):
    cap = cv2.VideoCapture(video)
    if index is None:
        index = int(round((t or 0) * cap.get(cv2.CAP_PROP_FPS)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(index, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)))
    ok, f = cap.read()
    if not ok:
        raise HTTPException(500, "cannot read frame")
    return f


def b64png(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


# ---------- API ----------

@app.get("/api/videos")
def api_videos():
    out = []
    for v in list_videos():
        out.append({"video": v, "analysed": os.path.exists(raw_path(v)),
                    "running": _jobs.get(v, {}).get("state") == "running"})
    return out


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(".mp4"):
        raise HTTPException(400, "only .mp4 videos are supported")
    os.makedirs(os.path.join("data", "uploads"), exist_ok=True)
    dst = os.path.join("data", "uploads", name)
    with open(dst, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"video": dst.replace("\\", "/")}


@app.get("/api/info")
def api_info(video: str):
    check_video(video)
    setup = load_setup(video)
    return {**video_info(video), "setup": setup.__dict__, "analysed": os.path.exists(raw_path(video)),
            "needs_rerun": needs_rerun(video, setup)}


@app.get("/api/video")
def api_video(video: str):
    """The file itself, for playback in the browser (FileResponse answers range requests, so seeking works)."""
    check_video(video)
    return FileResponse(video, media_type="video/mp4")


@app.get("/api/frame")
def api_frame(video: str, t: float = 0.0):
    check_video(video)
    ok, jpg = cv2.imencode(".jpg", frame_at(video, t), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(jpg.tobytes(), media_type="image/jpeg")


@app.post("/api/geometry")
def api_geometry(body: dict):
    """Pixel outlines for drawing the setup: outline, grid lines, zones and grid cells (open field) or the
    maze arms and its 12 points / 4 tips (plus maze)."""
    s = setup_from_body(body)
    if s.test == "epm":
        if len(s.maze_tips_px) != 4 and not s.maze_points_px:
            raise HTTPException(400, "maze tips missing")
        calib = s.calibration()
        zones = s.zones(calib=calib)
        px = lambda pts: calib.to_px(pts).round(1).tolist()  # noqa: E731
        points = s.maze_points()
        return {"outline": [points[p] for p in epm.POINTS], "grid": [], "cells": [],
                "zones": {name: [px(p) for p in polys] for name, polys in zones.drawing().items()},
                "points": points, "tips": epm.tips_from_points(points)}
    calib, zones = s.calibration(), s.zones(CFG["open_field"]["corner_fraction"])
    px = lambda pts: calib.to_px(pts).round(1).tolist()  # noqa: E731
    ncol, nrow = zones.grid_shape
    g = zones.grid_cm
    cells = []
    for r in range(nrow):
        for c in range(ncol):
            x2, y2 = min((c + 1) * g, zones.width_cm), min((r + 1) * g, zones.height_cm)
            cells.append({"col": c, "row": r, "poly": px([[c * g, r * g], [x2, r * g], [x2, y2], [c * g, y2]])})
    return {
        "outline": calib.corners_px.round(1).tolist(),
        "grid": [px(line) for line in zones.grid_lines_cm()],
        "zones": {name: [px(p) for p in polys] for name, polys in zones.drawing().items()},
        "cells": cells,
    }


@app.post("/api/setup")
def api_save_setup(video: str, body: dict):
    check_video(video)
    s = setup_from_body(body)
    s.corners_source = "saved setup of this video"
    rerun = needs_rerun(video, s)  # before saving: older detections take the previously saved setup as theirs
    s.save(os.path.join(state_dir(video), "setup.json"))
    return {"saved": True, "needs_rerun": rerun}


def _run_detector(video, setup):
    job = _jobs[video]
    try:
        from src.detector import MouseDetector
        dc = CFG["detector"]
        calib = setup.calibration()
        det = MouseDetector(dc["weights"], calib, imgsz=dc["imgsz"], conf=dc["conf"])

        def progress(i, total):
            job.update(done=i, total=total)

        raw = track_video(video, det, calib, batch=dc["batch"], progress=progress)
        raw.to_csv(raw_path(video), index=False)
        setup.save(detected_setup_path(video))
        job.update(state="done", done=job.get("total", 0))
    except Exception as e:  # report to the UI instead of dying silently
        job.update(state="error", error=str(e))


@app.post("/api/run")
def api_run(video: str, body: dict):
    check_video(video)
    with _lock:
        busy = [v for v, j in _jobs.items() if j.get("state") == "running"]
        if busy:
            return {"started": False, "reason": "already running" if video in busy else
                    f"{os.path.basename(busy[0])} is being analysed; one video at a time, try again when it is done."}
        setup = setup_from_body(body)
        setup.save(os.path.join(state_dir(video), "setup.json"))
        _jobs[video] = {"state": "running", "done": 0, "total": video_info(video)["frames"], "started": time.time()}
    threading.Thread(target=_run_detector, args=(video, setup), daemon=True).start()
    return {"started": True}


@app.get("/api/status")
def api_status(video: str):
    job = _jobs.get(video)
    if not job:
        return {"state": "done" if os.path.exists(raw_path(video)) else "idle"}
    out = dict(job)
    if job["state"] == "running" and job.get("done"):
        el = time.time() - job["started"]
        out["eta_s"] = round(el / job["done"] * (job["total"] - job["done"]))
    return out


def _analyse(video, setup):
    if not os.path.exists(raw_path(video)):
        raise HTTPException(409, "This video has not been analysed yet. Press 'Run analysis' first.")
    raw = pd.read_csv(raw_path(video))
    try:
        return analyze(raw, video_info(video)["fps"], setup, CFG)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/analyze")
def api_analyze(video: str, body: dict):
    check_video(video)
    setup = setup_from_body(body)
    res = _analyse(video, setup)
    pf = res["per_frame"]
    frame = frame_at(video, index=int(pf.frame.iloc[len(pf) // 2]))
    whole, zone_header, zone_rows = report.measures_tables(res)
    zone_cols = [c for c in pf.columns if c.startswith("in_")]
    px = lambda s: [None if pd.isna(v) else round(float(v), 1) for v in s]  # noqa: E731
    return JSONResponse({
        # per-frame track of the analysed window, for the playback overlay on the Results tab
        "track": {"t": [round(float(v), 3) for v in pf.time_s], "x": px(pf.x_px), "y": px(pf.y_px),
                  "moving": pf.moving.astype(bool).tolist(), "zone_names": [c[3:] for c in zone_cols],
                  "zones": pf[zone_cols].astype(bool).to_numpy().tolist()},
        "auto_start_s": res["auto_start_s"], "auto_end_s": res["auto_end_s"],
        "summary": {k: nan_safe(v) for k, v in res["summary"].items()},
        "whole": whole, "zone_header": zone_header, "zone_rows": zone_rows,
        "per_bin": res["per_bin"].to_dict(orient="records"),
        "images": {"track": b64png(report.png(report.track_figure(frame, res))),
                   "heatmap": b64png(report.png(report.heatmap_figure(res))),
                   "per_minute": b64png(report.png(report.per_minute_figure(res)))},
    })


@app.get("/api/export")
def api_export(video: str, kind: str):
    check_video(video)
    setup = load_setup(video)
    res = _analyse(video, setup)
    stem = os.path.splitext(os.path.basename(video))[0]
    if kind == "csv":
        return Response(report.summary_csv(stem, setup, res), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_results.csv"'})
    if kind == "track":
        return Response(report.track_csv(res), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_track.csv"'})
    if kind == "pdf":
        pf = res["per_frame"]
        frame = frame_at(video, index=int(pf.frame.iloc[len(pf) // 2]))
        return Response(report.pdf_bytes(stem, setup, frame, res), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_report.pdf"'})
    raise HTTPException(400, "kind must be csv, track or pdf")


@app.get("/api/validation")
def api_validation():
    d = os.path.join("outputs", "validation", "mouse_det_v2")
    pv = pd.read_csv(os.path.join(d, "per_video.csv"))
    ev = pd.read_csv(os.path.join(d, "engine_vs_reference.csv"))

    def stats(a, b):
        diff = a - b
        return {"mean_abs": round(float(diff.abs().mean()), 1),
                "mean_abs_pct": round(float((diff.abs() / b).mean() * 100), 1),
                "max_abs_pct": round(float((diff.abs() / b).max() * 100), 1),
                "r": round(float(np.corrcoef(a, b)[0, 1]), 3)}

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (a, b, title, unit) in zip(axes, [("ours_raw_distance_cm", "ref_raw_distance_cm", "Distance", "cm"),
                                              ("ours_center_time_s", "ref_center_time_s", "Centre time", "s")]):
        ax.scatter(pv[b], pv[a], color="#1f6fd1")
        lo, hi = min(pv[a].min(), pv[b].min()), max(pv[a].max(), pv[b].max())
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1)
        ax.set_xlabel(f"DLCAnalyzer ({unit})")
        ax.set_ylabel(f"Ours ({unit})")
        ax.set_title(title)
    fig.tight_layout()

    cols = ["video", "detected_pct", "err_median_cm", "ref_raw_distance_cm", "ours_raw_distance_cm",
            "ref_center_time_s", "ours_center_time_s", "ref_center_transitions", "ours_center_transitions"]
    return {
        "table": pv[cols].to_dict(orient="records"),
        "method": {
            "distance": stats(pv.ours_raw_distance_cm, pv.ref_raw_distance_cm),
            "centre_time": stats(pv.ours_center_time_s, pv.ref_center_time_s),
            "transitions": stats(pv.ours_center_transitions, pv.ref_center_transitions),
        },
        "engine": {
            "distance": stats(ev.ours_distance, ev.ref_distance),
            "centre_time": stats(ev.ours_centre_s, ev.ref_centre_s),
        },
        "frames": {"detected_pct": round(float(pv.detected_pct.mean()), 1),
                   "median_error_cm": round(float(pv.err_median_cm.median()), 2)},
        "scatter": b64png(report.png(fig)),
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "app", "static")), name="static")
