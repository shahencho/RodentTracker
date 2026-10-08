# RodentTracker

Software-only video tracking and analysis for lab rodent behaviour tests (an ANY-maze / EthoVision alternative). See [detect_mouse/Lab_Rodent_Behavior_Handoff.md](detect_mouse/Lab_Rodent_Behavior_Handoff.md) for the aim, the market and the validation plan.

## Setup (Windows)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Datasets are described in [data/README.md](data/README.md).

## Layout

| Path | Purpose |
|---|---|
| `src/` | tracking, zones, metrics |
| `scripts/` | entry points |
| `data/` | videos and answer keys (gitignored) |
| `outputs/` | tracks, metrics CSVs, reports (gitignored) |
