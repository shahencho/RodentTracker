# Elevated plus maze (EPM): plan

*Started 9 October 2026. Second test of the v1 scope (the first is the open field).*

## Goal

The same flow as the open field, for the elevated plus maze: one mouse, a video from a PC, no GPU. The user marks the maze, the tool tracks the mouse and reports arm times and entries. The results are checked against the DLCAnalyzer analysis of the Sturman et al. EPM videos.

## The test

- **Shape:** a "+" shaped maze raised above the floor.
- **Arms:** two **closed** arms with walls, two **open** arms without, and a **centre** square where they cross.
- **What it measures:** mice avoid the open arms. Less open-arm time and fewer open-arm entries mean more anxiety-like behaviour.
- **Length:** usually 5–10 minutes.

## Decisions (9 Oct)

| Topic | Decision |
|---|---|
| Marking the maze | **Both modes:** (1) the **4 arm tips plus the arm width**, which is simple; (2) the **12 corners** of the "+", which is accurate when the camera is tilted. Zones are built automatically, and every point can be dragged |
| Arm entry rule | **Animal's centre** (as in DLCAnalyzer, our reference). An ANY-maze-style "% of body in the arm" option comes later, because it needs the mouse outline |
| Head dips | Not in this version: they need head points (stage 2, pose) |
| YSMU EPM videos | None yet, so validation uses Sturman only |
| Reference | DeepLabCut tracks plus the DLCAnalyzer `EPMAnalysis` (the answer key), ported to Python like the open field |

## Reference data (already on disk, `data/dlcanalyzer/`)

- **DLC tracks for all 24 EPM videos** (`data/EPM/Output_DLC/`):
  - body points (nose, head, neck, ears, **body centre**, hips, tail);
  - **12 maze points**: closed-arm ends `tl tr` (top) and `bl br` (bottom), open-arm ends `lt lb` (left) and `rt rb` (right), centre corners `ctl ctr cbl cbr`.
- **Zone template** `example/EPM/EPM_zoneinfo.csv`: which points span each zone:
  - centre = `ctr ctl cbl cbr`;
  - closed top = `tl tr ctr ctl`, closed bottom = `cbr br bl cbl`;
  - open left = `lt ctl cbl lb`, open right = `rt rb cbr ctr`.
- **The DLCAnalyzer EPM pipeline** (README):
  1. read at 25 fps;
  2. cut the first 100 and last 250 frames;
  3. calibrate on the distance `tl`–`br` = **65.5 cm**;
  4. build zones from the median maze points;
  5. clean the track: likelihood below 0.95, or **outside the maze outline scaled 1.8×**, becomes missing and is interpolated;
  6. `EPMAnalysis`: moving above 5 cm/s with ±5-frame smoothing; zone reports for centre, open, closed and each arm.
- **Published check numbers** (example `EPM_1`): raw distance 1768.01 cm, distance moving 898.77 cm, time moving 97.88 s.
- **Human scoring** by three raters (Jin, Oliver, Sian): head dips, grooming, rearing and stretches. Not used now; useful for stage 2.
- **Possible extra reference:** the Zenodo set is described as videos "for DeepLabCut, Noldus EthoVision X14 and TSE … comparisons". If the paper's supplement gives EthoVision results per video, that is a commercial-tool reference. To check.

## Geometry

- Maze size: **tip-to-tip length** (default 65.5 cm, Sturman) and **arm width** (default 5.5 cm). Each arm is then (65.5 − 5.5) / 2 = 30 cm long.
- The 12 points always define the maze.
  - In "4 tips" mode they are generated from the tips and the width.
  - In "12 corners" mode the user drags each one.
- Pixel-to-cm conversion: a homography fitted from the 12 image points to the ideal "+" template in cm. It corrects a tilted camera, just as the 4 floor corners do in the open field.
- Zones in cm follow the user's 12 points mapped through the homography, so the zone shapes match the clicked points exactly.

## Measures

ANY-maze style names and DLCAnalyzer rules.
- **Whole test:** duration, % frames tracked, total distance, distance while moving, average and maximum speed, time moving, time immobile (at least 2 s), immobile episodes.
- **Per zone** (open arms, closed arms, centre, and each of the 4 arms): time, %, entries, exits, latency to first entry, distance in zone, average visit.
- **Main EPM indices:**
  - **% time in open arms** = open / (open + closed);
  - **% open-arm entries** = open entries / (open + closed entries);
  - total arm entries.
- **Pictures:** track plot, heat map, per-minute chart. **Downloads:** CSV and PDF.

## Steps

1. **Data.**
   - Download the 24 Sturman EPM videos.
   - Check the paper supplement for per-video EthoVision results.
2. **Reference port.**
   - Add `EPMAnalysis` to `experiments/dlcanalyzer_ref.py`.
   - It must reproduce the README `EPM_1` numbers exactly.
3. **Detector check.**
   - Run the current model (v2) on EPM frames against the DLC body centre.
   - If it isn't good enough, adapt it as before: YOLOE auto-labels on about 6 EPM videos, fine-tune v3 on open field plus EPM, and test on the rest.
4. **Engine.**
   - `src/calibration.py`: a point-set homography (12 points).
   - `src/zones.py`: polygon zones for the EPM.
   - `src/metrics.py` / `src/analysis.py`: the EPM measures, with the test type stored in the setup.
5. **UI.**
   - Test selector (open field / plus maze).
   - EPM handles: 4 tips with a width field, or 12 corners, switchable.
   - Length and width fields.
   - EPM results table and indices.
   - EPM section on the Validation page.
6. **Validation.**
   - On the Sturman EPM videos not used for training: our track against the DLC track through the same DLCAnalyzer code, plus the full product engine against the reference.
   - Report open-arm time, closed-arm time, centre time, entries and distance.
7. **Docs and checks.**
   - Update `docs/project_status.md`.
   - Extend `experiments/ui_check.py` to the EPM.
   - Commit.

## Open points

- Whether an EthoVision per-video reference exists (step 1).
- How many EPM videos the detector needs for adaptation (step 3).

## Progress log

**9 Oct**
- **Step 1.**
  - All 24 EPM videos downloaded (1280×960, 25 fps, about 10 minutes each).
  - The paper supplement has no per-video EthoVision numbers, only plots. EthoVision tracked the EPM, and the paper reports no significant DLC vs EthoVision differences in open-arm time or distance. Our reference stays DLC + DLCAnalyzer.
- **Step 2.** `experiments/dlcanalyzer_ref.py` reproduces the README `EPM_1` numbers exactly: raw distance 1768.008 cm, distance moving 898.768 cm, time moving 97.88 s.
- **Step 3.** The current detector (v2) finds the EPM mouse **without adaptation**: median 0.4–0.5 cm from the DLC body centre on EPM_17, 19 and 22. It saw no EPM video in training, so all 24 videos are unseen test data.
- **Box choice (found on EPM_17).** This setup has many dark, mouse-like objects (13–17 boxes per frame), and taking the most confident box jumped between the mouse and fixtures: distance came out +156%. New rule (`src/tracking.py`, `BoxChooser`):
  1. only boxes inside the apparatus region count;
  2. a box scores higher the more it differs from the empty-scene background (the mouse moves, fixtures don't);
  3. a soft penalty applies if the mouse couldn't have reached the box.

  Distance is now within 1.5% on EPM_17. All candidate boxes are saved, so the rules can be replayed without the detector (`experiments/epm_replay.py`).
- **Still open after one video:** open-arm time is −23 s against the reference (EPM_17). Most disagreements are frames where the mouse straddles the open-arm / centre border, with our point and DLC's ~0.3 cm apart on opposite sides of it. A 5.5 cm arm makes single-point zone scoring very sensitive. The full 24-video validation will show whether this evens out or is a bias.
- **Automatic start and end.** The hand threshold is now relative to each video's typical foreground (5× the median, at least 2,500 px). The fixed 2,500 px was below a mouse alone in the large EPM crop. YSMU times are unchanged.
- **Steps 4–5 built:**
  - engine: `src/epm.py` (12-point homography, zones, measures);
  - UI: test selector, 4-tips / 12-corners modes, length and width.

  Checked in the browser: `experiments/ui_check_epm.py` 12/12, and the open-field check 20/20.
