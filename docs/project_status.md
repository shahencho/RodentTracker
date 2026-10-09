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
- **Focus now:** a demo that reproduces third-party reference results. Licensing is deferred until we go to market.
- **Main open decisions** (details in section 5):
  1. The YSMU arena size and the lab's exact metric definitions.
  2. What to build next: plus maze, PDF report or UI.

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
- Git history: setup and detector work in `e2ef79a`; stage 1 and validation in the next commit.

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

## 5. Questions to answer before moving forward

### A. YSMU data and lab protocol
1. **What is the real inner floor size of the YSMU arena (cm)?** All YSMU distances and speeds scale directly with it.
2. **Are the floor corners right?** Please confirm them or click them with `scripts/calibrate.py`. Is the camera fixed between sessions, so one calibration covers all videos from that cage?
3. **How does the lab define the centre zone?** For example, the central 50% of the side (our default, same as DLCAnalyzer) or a fixed size in cm.
4. **Which analysis window?** For example, the full 5 minutes after the mouse is placed, or the first N minutes. Should the start be automatic (after the hand leaves, as now) or set by hand?
5. **Which metric definitions should we match?** DLCAnalyzer's (validated), the lab's current manual scoring, or ANY-maze's? This decides the default movement threshold and the zone-entry rule.
6. Can YSMU provide a few videos with **their own manual or ANY-maze scores**, for an independent check on their setup?

### B. Product and scope
7. **What next:** plus maze (the other v1 test), the PDF validation report, or the upload-and-click UI?
8. Which **outputs** does the lab need: CSV only, PDF per animal, per-group summaries, heat maps?
9. **Where does it run:** a desktop app on the lab PC, a local web page, or a hosted service? Is ~1× real-time CPU speed acceptable, or do we need GPU or cloud options?
10. **How should setting up a new lab work?** Today a new setup needs auto-labelling and fine-tuning, a few CPU hours, which we run. Do we offer that as a service, or try to build a general model (more labs' data in training) so most labs need no setup?

### C. Licensing and business — *deferred*
Decided 9 Oct: licensing (Ultralytics AGPL) and hardware advice wait until we go to the real market. **Current focus: a demo that reproduces the third-party reference results** (DLCAnalyzer and other published answer keys).

### D. Housekeeping — *done 9 Oct*
- Stage 1 code, validation scripts and this document are committed.
- The other session's manual box-review scripts were removed; the pipeline doesn't use them, and they can be recovered from commit `e2ef79a`.
- `.venv-dlc` (DeepLabCut) was deleted.
