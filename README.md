# RodentTracker

Software-only video tracking and analysis for lab rodent behaviour tests (an ANY-maze / EthoVision alternative).
One mouse per video, a normal PC, no GPU. Two tests so far: the **open field** and the **elevated plus maze (EPM)**.
See [detect_mouse/Lab_Rodent_Behavior_Handoff.md](detect_mouse/Lab_Rodent_Behavior_Handoff.md) for the aim, the
market and the validation plan.

Online demo (with a login): **https://rodent.alarmius.duckdns.org**. How it is hosted: [deploy/README.md](deploy/README.md).

## Status and plans

| Doc | What it covers |
|---|---|
| [docs/project_status.md](docs/project_status.md) | where the project stands, open-field validation results |
| [docs/epm_plan.md](docs/epm_plan.md) | plus maze: plan, validation progress and how to resume it |
| [docs/anymaze_parity_plan.md](docs/anymaze_parity_plan.md) | ANY-maze features we have, will build next, or leave out (the roadmap) |
| [docs/phase_a_navigation_plan.md](docs/phase_a_navigation_plan.md) | Phase A: the ANY-maze style tabs and planned-feature placeholders |

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

The tabs follow ANY-maze's order:

- **Protocol**: pick a video and the test (open field or plus maze), check or drag the arena points (4 floor
  corners, or the plus-maze arm points), set the arena size, grid and centre zone or arm sizes, adjust start/end,
  then "Run analysis" (once per video; later changes recalculate instantly).
- **Tests**: one row per video with its status; click a row to open it.
- **Results**: tables, track plot, heat map, per-minute chart, CSV and PDF downloads.
- **Validation**: our numbers vs the DLCAnalyzer reference on the Sturman videos.
- **Experiment**, **Data** and **Roadmap**: planned features, shown as disabled placeholders where ANY-maze users
  would look for them. The **Show planned features** switch in the header hides them for a clean demo.

`scripts/seed_ui_state.py` pre-loads the YSMU and Sturman setups (open field and plus maze) and cached detections,
so the demo opens with results. The command-line equivalent is `scripts/analyze_video.py`.

## Layout

| Path | Purpose |
|---|---|
| `src/` | calibration, detector, tracking, zones, metrics (open field and EPM), analysis, report (plots, CSV, PDF) |
| `app/` | demo UI: FastAPI server + one static HTML/JS page; planned features listed in `app/static/planned.js` |
| `deploy/` | online demo on the droplet: setup, deploy, pm2, nginx, demo data list |
| `scripts/` | entry points |
| `experiments/` | validation against reference answers, detector probes and training, headless UI checks |
| `calib/` | saved arena calibrations (corners and real size) for the lab and Sturman videos |
| `docs/` | status, plans and roadmap (see above) |
| `research_notes/` | notes on pretrained rodent detectors, pose models and related papers |
| `detect_mouse/` | the project handoff document |
| `models/` | trained mouse detector weights (gitignored) |
| `data/` | videos and answer keys (gitignored) |
| `outputs/` | tracks, metrics CSVs, reports (gitignored) |
