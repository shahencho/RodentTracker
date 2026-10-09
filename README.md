# RodentTracker

Software-only video tracking and analysis for lab rodent behaviour tests (an ANY-maze / EthoVision alternative). See [detect_mouse/Lab_Rodent_Behavior_Handoff.md](detect_mouse/Lab_Rodent_Behavior_Handoff.md) for the aim, the market and the validation plan.

## Setup (Windows)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Datasets are described in [data/README.md](data/README.md).

## Demo UI

```
run_ui.bat          # starts the local server and opens http://127.0.0.1:8765/
```

Setup tab: pick a video, check or drag the 4 floor corners, set the arena size, grid and centre zone (or click
grid squares), adjust start/end, then "Run analysis" (once per video; later changes recalculate instantly).
Results tab: tables, track plot, heat map, per-minute chart, CSV and PDF downloads. Validation tab: our numbers
vs the DLCAnalyzer reference on 14 Sturman videos.

`scripts/seed_ui_state.py` pre-loads the YSMU and Sturman setups and cached detections, so the demo opens with
results. The command-line equivalent is `scripts/analyze_video.py`.

Online demo (DigitalOcean droplet, with a login): [deploy/README.md](deploy/README.md).

## Layout

| Path | Purpose |
|---|---|
| `src/` | calibration, detector, tracking, zones, metrics, analysis, report (plots, CSV, PDF) |
| `app/` | demo UI: FastAPI server + one static HTML/JS page |
| `deploy/` | online demo on the droplet: setup, deploy, pm2, nginx, demo data list |
| `scripts/` | entry points |
| `data/` | videos and answer keys (gitignored) |
| `outputs/` | tracks, metrics CSVs, reports (gitignored) |
