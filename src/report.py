"""Pictures, CSV and PDF for one analysed video (used by the CLI and the UI)."""
import io
from datetime import datetime

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ZONE_COLOURS = {"centre": "#1f6fd1", "corners": "#e07b00", "open": "#d62020", "closed": "#2e9d4f"}
LABELS = {
    "test_duration_s": ("Test duration", "s"), "frames_tracked_pct": ("Frames tracked", "%"),
    "open_time_pct": ("Time in open arms", "% of arm time"), "open_entries_pct": ("Open-arm entries", "% of arm entries"),
    "arm_entries": ("Total arm entries", ""),
    "total_distance_cm": ("Total distance travelled", "cm"), "distance_moving_cm": ("Distance while moving", "cm"),
    "average_speed_cm_s": ("Average speed", "cm/s"), "max_speed_cm_s": ("Maximum speed", "cm/s"),
    "time_moving_s": ("Time moving", "s"), "time_immobile_s": ("Time immobile", "s"),
    "immobile_episodes": ("Immobile episodes", ""), "line_crossings": ("Line crossings", ""),
}
ZONE_LABELS = {"time_s": "Time (s)", "time_pct": "Time (%)", "entries": "Entries", "exits": "Exits",
               "latency_s": "Latency to first entry (s)", "distance_cm": "Distance in zone (cm)",
               "avg_visit_s": "Average visit (s)"}


def _hex_to_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def overlay(frame, result, path=True):
    """Video frame with the floor outline, grid, zones and (optionally) the mouse path. Returns a BGR image."""
    vis = frame.copy()
    calib, zones = result["calibration"], result["zone_spec"]
    for a, b in zones.grid_lines_cm():
        p = calib.to_px([a, b]).astype(int)
        cv2.line(vis, tuple(p[0]), tuple(p[1]), (170, 170, 170), 1)
    for name, polys in zones.drawing().items():
        for poly in polys:
            cv2.polylines(vis, [calib.to_px(poly).astype(np.int32)], True, _hex_to_bgr(ZONE_COLOURS[name]), 2)
    cv2.polylines(vis, [calib.corners_px.astype(np.int32)], True, (0, 0, 220), 2)
    if path:
        pf = result["per_frame"][["x_cm", "y_cm"]].dropna().to_numpy()
        if len(pf):
            cv2.polylines(vis, [calib.to_px(pf).astype(np.int32)], False, (0, 170, 0), 1)
    return vis


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def track_figure(frame, result):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cv2.cvtColor(overlay(frame, result), cv2.COLOR_BGR2RGB))
    ax.set_title("Track plot", fontsize=10)
    ax.axis("off")
    return fig


def heatmap_figure(result, bin_cm=1.0):
    zones = result["zone_spec"]
    x0, x1, y1, y0 = zones.extent  # extent is (left, right, bottom, top)
    pf = result["per_frame"].dropna(subset=["x_cm"])
    x = np.clip(pf.x_cm, x0, x1 - 1e-6)
    y = np.clip(pf.y_cm, y0, y1 - 1e-6)
    h, _, _ = np.histogram2d(y, x, bins=[np.arange(y0, y1 + bin_cm, bin_cm), np.arange(x0, x1 + bin_cm, bin_cm)])
    fps = len(result["per_frame"]) / max(result["summary"]["test_duration_s"], 1e-9)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(np.log1p(h / fps), extent=list(zones.extent), cmap="magma")
    for name, polys in zones.drawing().items():
        for poly in polys:
            p = np.array(list(poly) + [poly[0]])
            ax.plot(p[:, 0], p[:, 1], color=ZONE_COLOURS[name], lw=1.5)
    ax.set_title("Heat map (time spent, log scale)", fontsize=10)
    ax.set_xlabel("cm")
    ax.set_ylabel("cm")
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + seconds)")
    return fig


def per_minute_figure(result):
    pb = result["per_bin"]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    minutes = pb.bin_start_s / 60
    ax.bar(minutes, pb.distance_cm, width=0.8, color="#7a9cc6", label="Distance (cm)", align="edge")
    ax.set_xlabel("Minute of test")
    ax.set_ylabel("Distance (cm)")
    ax2 = ax.twinx()
    col, label, colour = (("open_time_s", "Open-arm time (s)", ZONE_COLOURS["open"]) if "open_time_s" in pb
                          else ("centre_time_s", "Centre time (s)", ZONE_COLOURS["centre"]))
    ax2.plot(minutes + 0.4, pb[col], "o-", color=colour, label=label)
    ax2.set_ylabel(label)
    ax.set_title("Per minute", fontsize=10)
    fig.legend(loc="upper right", fontsize=8)
    return fig


def settings_text(setup):
    if setup.test == "epm":
        return (f"Elevated plus maze {setup.arm_span_cm:g} cm tip to tip, arms {setup.arm_width_cm:g} cm wide; "
                f"maze marked by {'4 arm tips' if setup.maze_mode == 'tips' else '12 corners'}.")
    centre = (f"{setup.center_value:g}% of side" if setup.center_mode == "percent"
              else f"{setup.center_value:g} cm from walls") + (
        f" + {len(setup.toggled_cells)} toggled squares" if setup.toggled_cells else "")
    return (f"Open field {setup.width_cm:g} x {setup.height_cm:g} cm, grid {setup.grid_cm:g} cm. Centre zone: {centre}. "
            f"Corner zones: {'on (20% of side)' if setup.corner_zones else 'off'}.")


def summary_row(video, setup, result):
    """One row per animal: settings + whole-test measures + zone measures (zone_measure columns)."""
    row = {"video": video, "test": setup.test, "settings": settings_text(setup), **result["summary"]}
    for zone, r in result["zones"].iterrows():
        for k, v in r.items():
            row[f"{zone}_{k}"] = v
    return row


def summary_csv(video, setup, result):
    return pd.DataFrame([summary_row(video, setup, result)]).to_csv(index=False)


def track_csv(result):
    return result["per_frame"].round(3).to_csv(index=False)


def measures_tables(result):
    s = result["summary"]
    whole = [(LABELS[k][0], f"{s[k]:g} {LABELS[k][1]}".strip()) for k in LABELS if k in s]
    whole.insert(0, ("Analysed", f"{s['start_s']:.1f} s to {s['end_s']:.1f} s"))
    zt = result["zones"]
    zone_rows = [[ZONE_LABELS[k]] + ["–" if pd.isna(zt.loc[z, k]) else f"{zt.loc[z, k]:g}" for z in zt.index]
                 for k in ZONE_LABELS]
    names = {"open": "Open arms", "closed": "Closed arms", "open_left": "Open left", "open_right": "Open right",
             "closed_top": "Closed top", "closed_bottom": "Closed bottom"}
    return whole, ["Measure"] + [names.get(z, z.capitalize()) for z in zt.index], zone_rows


def pdf_bytes(video, setup, frame, result):
    whole, zone_header, zone_rows = measures_tables(result)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        title = "Elevated plus maze report" if setup.test == "epm" else "Open field report"
        fig.text(0.08, 0.95, title, fontsize=18, weight="bold")
        fig.text(0.08, 0.925, f"Video: {video}", fontsize=10)
        fig.text(0.08, 0.908, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", fontsize=10)
        settings = (settings_text(setup) + "\n"
                    "Rules: DLCAnalyzer (moving > 5 cm/s, +-5 frame smoothing); immobile = not moving for >= 2 s; "
                    "zone entry by the animal's centre.")
        fig.text(0.08, 0.87, settings, fontsize=8.5, wrap=True)
        ax = fig.add_axes([0.08, 0.5, 0.84, 0.33])
        ax.axis("off")
        ax.set_title("Whole test", loc="left", fontsize=11)
        t = ax.table(cellText=[[a, b] for a, b in whole], colWidths=[0.6, 0.4], loc="upper left")
        t.scale(1, 1.4)
        ax = fig.add_axes([0.08, 0.12, 0.84, 0.33])
        ax.axis("off")
        ax.set_title("Zones", loc="left", fontsize=11)
        t = ax.table(cellText=zone_rows, colLabels=zone_header, loc="upper left")
        t.scale(1, 1.4)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        for rect, make in [([0.05, 0.55, 0.5, 0.38], lambda: overlay(frame, result))]:
            ax = fig.add_axes(rect)
            ax.imshow(cv2.cvtColor(make(), cv2.COLOR_BGR2RGB))
            ax.set_title("Track plot", fontsize=10)
            ax.axis("off")
        for rect, figfn in [([0.55, 0.55, 0.42, 0.38], heatmap_figure), ([0.08, 0.12, 0.84, 0.33], per_minute_figure)]:
            sub = figfn(result)
            img = plt.imread(io.BytesIO(png(sub)))
            ax = fig.add_axes(rect)
            ax.imshow(img)
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
