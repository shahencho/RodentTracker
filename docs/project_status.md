# RodentTracker — project status

*Last updated: 9 October 2026*

## Where we are now

**Stage 1 (single-mouse tracking and open-field metrics) works and has been validated.**

- **Pipeline.** The lab supplies a video and the 4 floor corners of the arena with its real size. They get a cleaned mouse path in cm, distance, speed, time moving, centre time and centre entries, a per-minute table and a path plot. It runs on a normal CPU (no GPU), at roughly real time: a 10-minute video takes about 12–13 minutes.
- **Accuracy (validated).** We tested on 14 public Sturman et al. open-field videos that the model never saw during training, against the published DLCAnalyzer answer key:

  | Metric | Average difference from the answer key | Worst video | Correlation |
  |---|---|---|---|
  | Distance travelled | 1.1% | 2.4% | r = 0.996 |
  | Centre time | 1.6 s (2.4%) | 6.1 s | r = 0.998 |
  | Centre transitions | 3.7 (6.9%) | | r = 0.969 |
  | Time moving | +12 s (5%) | | r = 0.989 |

  The mouse is found in 99.9% of frames, with a median position error of 0.49 cm.
- **YSMU videos.** Both videos are tracked in 100% of frames after the hand leaves the arena. The YSMU numbers in cm are not final yet, because the real arena size is unknown (40 × 40 cm is a placeholder).
- **Detector.** We reused existing models and did no hand labelling.
  - YOLOE (an open-vocabulary detector), given 4 example boxes, produced the training labels automatically.
  - A small public mouse model (KUNIN YOLO11n) was fine-tuned on those labels.
  - Each new lab setup needs a one-time adaptation of a few CPU hours, again with no hand labelling.
- **Focus now:** a simple demo UI on a PC (no GPU) for one mouse in the open field, modelled on ANY-maze and using the DLCAnalyzer-validated metric rules. Licensing is deferred until we go to market. All decisions are in section 5; the build steps are in section 7.
- **Waiting on YSMU:** which other tests they record, and the format of their reference scores (section 6).

---

## 1. Requirement (from the handoff)

Source: [detect_mouse/Lab_Rodent_Behavior_Handoff.md](../detect_mouse/Lab_Rodent_Behavior_Handoff.md)

**Goal.** Build software-only video tracking and analysis for rodent behaviour tests, as an alternative to ANY-maze or EthoVision. The lab uploads a video, picks the test, clicks the arena zones, and gets metrics, a CSV and a PDF report. The lab does no coding, needs no GPU and does no labelling.

**Market context.**
- The real price to beat in Armenia is ANY-maze's developing-countries licence at $950/year.
- Target labs that own no licence, and the cases where paid tools fail: dark or IR video, low contrast, complex behaviours.
- Pilot candidates are the Orbeli Institute and the Mndzhoyan Institute. YSMU supplied the sample videos.

**Model plan.**

| Stage | What | Metrics | Status |
|---|---|---|---|
| 1 | Detection + tracking of the body centre | distance, speed, zones, entries | **Done, validated on open field** |
| 2 | Rodent pose (nose, ears, tail base…) | head dips, object exploration | Not started |
| 3 | Temporal behaviour classifier | rearing, grooming, freezing, immobility | Not started |

**v1 scope.**
- In: stage 1 only, open field and plus maze, plus a validation report showing our numbers match reference answers.
- Out: live cameras, multiple animals, 24/7 home-cage recording.

**Next concrete step named in the handoff:** "run stage 1 on the Sturman open-field videos and compare distance and center time with the DLCAnalyzer answers". **This is now done**: see section 3.

---

## 2. What we did, in order

### 2.1 Set up the project
- Created the repo `C:\Dev_projects\RodentTracker`, separate from Shoplifting but reusing its patterns: its own `.venv` (Python 3.13, ultralytics 8.4, CPU-only PyTorch), `config.yaml`, and `src/`, `scripts/` and `experiments/` folders.
- DeepLabCut was installed in a separate venv for the model tests only; it has since been deleted.
- Git history: setup and detector work in `e2ef79a`; stage 1 and validation in `967c442`.

### 2.2 Looked at the YSMU sample (`cntrol60_cage22_an1/an2_OF1.mp4`)
- 640×480 grayscale, 19.35 fps (an1) and 24.95 fps (an2), about 5 minutes each.
- **This is the hard case:** a white mouse (about 30 px) in a white arena, filmed by a tilted camera with walls visible. The mouse often presses against the walls or rears, and the camera's auto-exposure drifts.
- In an2, the experimenter's gloved hand is in the arena for the first ~15 s.

### 2.3 Tested ready-made models on the YSMU video (no training)

Each model was scored on 120 frames spread over the video. A hit means within 30 px of the mouse.

| Model | Hit / 120 | Notes |
|---|---|---|
| YOLO11n / YOLO11m (COCO) | 0–7 | COCO has no rodent class; it labelled the mouse "vase" or "bowl" |
| YOLO-World s / x, text prompt "mouse" | 0–1 | |
| DeepLabCut SuperAnimal-TopViewMouse | 0–14 | Fails on a white mouse, ~9.5 s/frame, non-commercial licence |
| KUNIN mouse YOLO (Hugging Face, CC BY 4.0) | ~50–66 | Fast (~0.15 s/frame) but often off target |
| **YOLOE-11l, given example boxes of the mouse** | **120 (0 wrong)** | With 4 example boxes and confidence 0.01; ~1.3 s/frame, too slow to track every frame |

We also ran deep research on pretrained rodent models; the report is in [reports/Pretrained rodent YOLO models.md](../reports/Pretrained%20rodent%20YOLO%20models.md). Its conclusion: no downloadable model fits our footage, runs well on a CPU and is licensed for commercial use.

### 2.4 Chosen approach: reuse models to make labels, then fine-tune a small fast model
1. **Auto-labelling.** YOLOE was given 4 example boxes of the mouse (in the open, at a wall, in a corner) and run on every 5th frame of both YSMU videos.
2. **Filling gaps.** Frames YOLOE missed were filled from background subtraction and checked by eye.
3. **Fixes found along the way:**
   - The experimenter's hand was spoiling one example box, so frames with a hand in view are now detected automatically and skipped.
   - On an2, YOLOE missed the mouse when it was walking stretched out, which is what the background-subtraction fills covered.
4. **Fine-tuning.** We fine-tuned a YOLO11n starting from the KUNIN mouse backbone, on frames cropped to the arena (3× faster on the CPU). This is **v1**: 120/120 and 114/114 on the YSMU test frames, at about 50–70 ms per frame.

### 2.5 Built stage 1 (the product pipeline)

| File | Purpose |
|---|---|
| `src/calibration.py` | Converts pixels to cm from 4 floor corners and the real size, correcting for the tilted camera |
| `src/detector.py` | Runs the fine-tuned YOLO on the arena crop and keeps the most confident box |
| `src/tracking.py` | Runs every frame, rejects impossible jumps (> 100 cm/s), fills gaps ≤ 0.5 s, smooths the path, and detects the hand to start the analysis automatically |
| `src/metrics.py` | Open-field metrics: distance, speed, time moving, centre zone (sides at 0.5 of the arena), entries with a 0.2 s minimum, latency, per-minute bins |
| `scripts/calibrate.py` | Click the 4 floor corners (or pass them on the command line); writes a calibration file and a preview with a 5 cm grid |
| `scripts/analyze_video.py` | Video in; `track.csv`, `metrics.csv`, `bins.csv` and `track.png` out |
| `config.yaml` | All thresholds in one place |

**YSMU results.** These depend on the placeholder 40 × 40 cm arena size.

| Video | Analysed | Distance | Centre time | Centre entries |
|---|---|---|---|---|
| an1 | 0–309 s | 515 cm | 3.0 s | 1 |
| an2 | 14.9–313 s (after the hand left) | 683 cm | 2.3 s | 1 |

Both mice stayed close to the walls. Each crossed the centre zone once.

### 2.6 Validation on Sturman et al. (the handoff's next step)
- **Data.** 20 open-field videos from Zenodo 3608658 (CC-BY 4.0): 928×576, 25 fps, 10 minutes each, black C57 mice, 4 arenas, wide-angle lens. The answer key is the DLCAnalyzer repo (GPL-3.0).
- **Answer key.** DLCAnalyzer's open-field analysis was ported from R to Python (`experiments/dlcanalyzer_ref.py`). It **reproduces their published OFT_3 numbers exactly** (4008.34 cm, 79.8 s, 58 transitions).
- **The first model doesn't transfer to a new lab.** The YSMU-only model (v1) jumped between the black mouse and dark wall fixtures, and its distance came out 17 times too high.
- **Adapting to the new setup:**
  1. Auto-labelled 6 of the 20 videos, covering all 4 arenas, with YOLOE at one frame per second.
  2. Dropped the background-subtraction fills on this setup: they caught the black tail and empty frames at the end of recordings.
  3. Fine-tuned **v2** on YSMU and Sturman together (3.1 h on the CPU). v2 scores the same on YSMU as v1.
- **Test.** The other 14 videos, never seen in training. Our track and DeepLabCut's body-centre track both went through the same DLCAnalyzer code with the same corners, so differences come from tracking only. Results are in the summary at the top.
- **Output:** `outputs/validation/mouse_det_v2/per_video.csv`.

### 2.7 Key files and artefacts

| Item | Location |
|---|---|
| Current detector | `models/mouse_det_v2.pt` (v1: `models/mouse_det_v1.pt`) |
| Calibrations | `calib/ysmu_cage22.json`, `calib/sturman/OFT_*.json` |
| Auto-labels | `data/labels/yoloe/` |
| Training runs | `outputs/train/kunin_ft`, `outputs/train/mouse_det_v2` |
| Validation | `outputs/validation/mouse_det_v2/` |
| Experiment scripts | `experiments/` (probes, labelling, fine-tuning, validation) |

Data, outputs and weights are gitignored.

---

## 3. Limitations to keep in mind

- **The answer key is a machine reference.** It is DeepLabCut tracking analysed by DLCAnalyzer, not human scoring. The Sturman paper did validate it, but no human-scored check of our own output exists yet.
- **Only two labs tested so far** (YSMU and Sturman). Each new setup (camera, mouse colour, arena) is expected to need the one-time auto-labelling and fine-tuning step, a few CPU hours. This step isn't automated for end users yet.
- **The YSMU accuracy check is weaker.** It compares against background-subtraction positions checked by eye, not against an independent answer key.
- **Lens distortion is not corrected.** The 4-corner calibration assumes a flat, undistorted floor. Sturman's wide-angle lens bends the edges; it didn't affect the comparison because DLCAnalyzer makes the same simplification, but absolute distances on such lenses carry a small bias.
- **Metric definitions differ between tools.** The validated numbers use DLCAnalyzer's definitions: moving above 5 cm/s, smoothed over ±5 frames. Our default `analyze_video.py` uses its own: moving above 2 cm/s and a 0.2 s minimum zone stay. On YSMU, distance varied from 515 to 605 cm depending on the threshold. ANY-maze and EthoVision each use their own definitions too, which is part of why they disagree (Lim et al. 2023).
- **Speed.** Roughly real time on this laptop CPU. That's fine for batch use, but a lab with 50 videos waits about a day.
- Single animal, top-down camera, open field only so far.

---

## 4. Licences

| Component | Licence | Implication |
|---|---|---|
| Ultralytics YOLO (training and inference) | AGPL-3.0 | A closed or commercial product needs an Ultralytics Enterprise licence, or the model must be moved to another runtime or framework (e.g. ONNX export plus our own code — check whether exported weights stay covered) |
| KUNIN mouse model (our starting weights) | CC BY 4.0 | Commercial use allowed with attribution |
| YOLOE | AGPL-3.0 (Ultralytics) | Used only as an offline labelling tool |
| Sturman videos | CC BY 4.0 | Fine for validation; cite them |
| DLCAnalyzer | GPL-3.0 | Our Python port is a derivative, so keep it internal (validation only) or release it under GPL |
| DeepLabCut SuperAnimal | Non-commercial | Tested only; not used |

---

## 5. Decisions log

Every question raised so far, with the answer given (8–9 October 2026).

### A. YSMU data and lab protocol
| # | Question | Decision |
|---|---|---|
| A1 | Real inner floor size of the YSMU arena? | Use **40 × 40 cm** as the default; editable in the UI |
| A2 | Are the floor corners right? | Confirm them **in the UI** (not with `calibrate.py`). Corners come from the last saved setup for that camera and can be dragged. The UI must say clearly where the corners came from |
| A3 | How is the centre zone defined? | The UI shows the centre zone used and lets the user adjust it, because the YSMU camera is off-centre and tilted. **Default 50% of the arena side**; can also be set in cm from the walls, plus **ANY-maze-style click-to-toggle grid squares** |
| A4 | Analysis window? | **Start** when the hand has left and the mouse is in the arena. **End** when the hand returns or the video ends; recordings longer than 5 minutes are analysed to the end. Both times are auto-detected **and editable in the UI** |
| A5 | Which metric definitions? | **DLCAnalyzer's are the default** (they're what we validated against). No YSMU reference data exists yet to compare against |
| A6 | Can YSMU provide their own scores? | Expected soon; format not yet known |

### B. Product and scope
| # | Question | Decision |
|---|---|---|
| B7 | What to build next? | A **simple UI** showing what we do, with the YSMU results. Very simple; show the most important outputs; anything not covered comes later |
| B8 | Which outputs? | **CSV and PDF per animal.** All outputs are wanted (heat maps etc.); anything complicated can move to the next phase |
| B9 | Where does it run? | **PC only, no GPU.** A local web page that opens in the browser |
| B10 | How does a new lab get set up? | For now, a few CPU hours of auto-labelling and fine-tuning per setup. A general model is deferred |
| – | Per-group summaries / several animals? | Too complicated for now: **one animal per video.** Design the code so groups can be added later |
| – | UI language? | **English only** |
| – | Validation report: PDF or UI page? | A **read-only UI page** (the Sturman comparison) |
| – | Demo scope? | Open field only, very simple UI, the most important outputs |

### C. Licensing and business — *deferred*
| # | Question | Decision |
|---|---|---|
| C11 | Ultralytics AGPL route? | Deferred until we go to market; keep using the current stack |
| C12 | Hardware (IR kit)? | No hardware. Work with whatever YSMU records, and state clearly which of their tests we can't produce |

### D. Housekeeping — *done 9 Oct*
| # | Question | Decision |
|---|---|---|
| D13 | Commit stage 1? | Done (`967c442`) |
| D14 | Scripts from another session? | Not needed by the pipeline, so removed (recoverable from `e2ef79a`) |
| D15 | Delete the DeepLabCut venv? | Deleted |

### E. UI design (after the ANY-maze research)
| Question | Decision |
|---|---|
| How to define the centre zone | Default **50% of side**; or **cm from walls**; plus **click grid squares** to add or remove them |
| Corner zones | **Yes** (DLCAnalyzer definition: a square 20% of the side in each corner) |
| Line crossings | **Yes** (the mouse's centre moves from one grid square to another) |
| Immobility | **Yes, now:** not moving (DLCAnalyzer rule) for at least 2 s |
| Grid size | **10 cm default**, user-adjustable |
| Corner auto-detection | Corners come **from the previous saved setup**; the UI states this clearly |
| Video input | Pick from a folder list on the PC, plus an Upload button |
| Validation page | Read-only, from the results we already have |
| Zone entry point | The animal's **centre** (the same as ANY-maze's common setting and DLCAnalyzer) |

---

## 6. Still open

1. **YSMU's other tests:** which tests do they record besides open field (plus maze, Y-maze, forced swim, novel object…)? With sample videos we can say what we can and can't produce.
2. **YSMU reference data:** format (ANY-maze export, Excel of manual scores…) and which videos.
3. **Later phases:** groups and multiple animals, other tests, head and pose points (head entries, freezing, rearing), a general model, licensing.

---

## 7. Next stage: demo UI (open field, one mouse)

### 7.1 How ANY-maze does it (what we copy)
- **Setup:**
  - a background frame from the video;
  - a ruler line for scale;
  - the arena outline drawn on the frame;
  - a square grid at N cm;
  - zones made by **selecting grid squares**.
- **Typical zones:** centre, wall (thigmotaxis) and corners. ANY-maze doesn't fix a default centre size ("some researchers minimise it, while others define it as half of the overall test arena").
- **Zone entry rule:** the animal's centre, head, or a % of its body.
- **Start and end:** auto-start when the operator leaves the camera's view; end after a set duration or on an event.
- **Re-analysis:** zones can be changed before, during or after tests, and results recalculate.
- **Open-field measures:**
  - whole test: test duration, total distance, average and maximum speed, time mobile and immobile, immobile episodes, line crossings, % frames tracked;
  - per zone: time, %, entries, exits, latency to first entry, distance in zone, visit durations;
  - across time: any measure per time bin;
  - pictures: track plot and heat map;
  - export: CSV.
- **Calculation details:** distance uses "adaptive smoothing" scaled to the animal's size; maximum speed is measured over a minimum distance. Their exact algorithms aren't published, so we use the validated DLCAnalyzer rules instead.

Sources:
- [Setting up apparatus](https://www.any-maze.com/features/setting-up-apparatus/)
- [Running tests](https://www.any-maze.com/features/running-tests/)
- [Open field](https://www.any-maze.com/applications/open-field/)
- [Measures (PDF)](https://www.any-maze.com/wp-content/uploads/2023/01/a-detailed-description-of-the-any-maze-measures.pdf)
- [Open-field protocol using ANY-maze (protocols.io)](https://www.protocols.io/view/open-field-locomotion-test-cuy6wxze.html)

### 7.2 Screens
How it runs: a local web page started with `run_ui.bat`. A small Python server (FastAPI) reuses `src/`, and one plain HTML/JavaScript page draws on a canvas. No framework and no internet.

1. **Setup**
   - Pick a video (folder list or Upload).
   - A video frame with **4 draggable floor corners** and a notice saying where they came from, e.g. "copied from saved setup of <video> — drag to adjust" or "no saved setup — default corners".
   - Arena size (40 × 40 cm), grid (10 cm), centre zone (50% / cm from walls, click squares to toggle, "Reset to default"), corner zones on/off.
   - A timeline with auto start and end that can be edited.
   - "Run analysis" with a progress bar. The detector runs once per video; any later change recalculates instantly.
2. **Results**
   - Whole-test table, zone table (centre / wall / corners), track plot, heat map, per-minute chart.
   - Download **CSV** and a **2-page PDF**:
     - page 1: video, date, settings used and tables;
     - page 2: plots.
3. **Validation**
   - The method paragraph, the 14 Sturman videos (ours next to DLCAnalyzer's), scatter plots, and the average and worst differences.

### 7.3 Calculation rules
| Item | Rule |
|---|---|
| Position and zone entry | The centre of the mouse's box |
| Distance and speed | DLCAnalyzer rules (validated). Moving = faster than 5 cm/s, smoothed over ±5 frames |
| Immobility | Not moving for at least 2 s: time immobile and number of episodes |
| Centre zone | 50% of the side (default), or N cm from the walls, plus toggled grid squares |
| Wall zone | The arena minus the centre |
| Corner zones | A square 20% of the side wide in each corner (DLCAnalyzer) |
| Line crossings | The mouse's centre moves from one grid square to another (10 cm default) |
| Max speed | Measured over at least 2 cm of movement |
| Timing | Auto start (hand gone, mouse present) and auto end (hand returns or the video ends), both editable |

### 7.4 Step-by-step
0. **Update this document** with all decisions and the plan. *Done.*
1. **Metrics engine.**
   - Move the DLCAnalyzer rules into `src/metrics.py` as the default.
   - Add `src/zones.py` (centre / wall / corners as floor-plane masks, toggled grid squares, line crossings).
   - Add immobility episodes, max speed over at least 2 cm, and per-zone measures.
   - Add `auto_end_s()` in `src/tracking.py`.
   - Re-score the 14 Sturman videos from the cached tracks; the numbers must stay identical.
2. **Backend (`app/server.py`):**
   - list and upload videos;
   - serve frames;
   - save and load the setup per video, with the corner source rule (this video → last saved setup from the same camera → default);
   - run the detector in the background with progress;
   - instant re-analysis;
   - CSV and PDF export;
   - validation data.
3. **Setup screen:** draggable corners, the perspective grid, centre and corner zones, click-to-toggle squares, the corner-source notice, and timeline start and end.
4. **Results screen and exports:** tables, track plot, heat map and per-minute chart (`src/report.py`, matplotlib), CSV, and the 2-page PDF.
5. **Validation screen.**
6. **Launcher and docs:** `run_ui.bat`, README section, update this document, commit.

**Checks after each step:**
- the Sturman numbers are unchanged;
- YSMU an1 and an2 run end to end (an2 auto start ≈ 14.9 s);
- changing a zone updates results instantly;
- the CSV and PDF open.

Browser automation isn't connected, so the user checks the screens visually after each step.
