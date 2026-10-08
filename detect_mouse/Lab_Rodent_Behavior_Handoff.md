# Handoff: Lab rodent behavior analysis (ANY-maze alternative)

## Goal

Build a software-only tool that replicates the video tracking and analysis part of ANY-maze / EthoVision for research labs.

The lab uploads a video, picks the test type, clicks the arena zones, and gets metrics, a CSV and a PDF report. No coding, no GPU and no labelling on the lab's side.

## Competitors (verified)

- **ANY-maze** — <https://www.any-maze.com/pricing/>
  - Full licence $7,995 one-time, all features, lifetime updates and support.
  - Developing-countries licence $950/year; Armenia qualifies (educational or non-profit labs only).
  - TakeNote $650 (manual scoring only). I/O-only $2,195 (no video). Free 30-day trial.
  - Also sells cameras, computers, mazes and interface boxes.
- **Noldus EthoVision XT** — ~2,000+ sites, $3,495 / $9,995, extra paid modules (rearing, head dips and others), plus its own hardware.
- **Others** — ConductVision, Panlab SMART, and free DeepLabCut / SLEAP / SimBA, which are powerful but need coding and a GPU.

In Armenia, the real price to beat is **$950/year**, not $8,000.

## What real users complain about (ResearchGate, papers)

- EthoVision is hard to use and charges thousands for each module.
- Automatic scoring of complex behaviours (grooming, rearing, social) isn't accurate in either product and needs expert tuning.
- Tracking fails on low-contrast animals (black mouse on a dark floor), on truly dark or red-light video, and with water reflections.
- The two tools give different numbers on the same video (Lim et al. 2023, *J Neurosci Methods*), so validation matters.
- Budget labs say outright that ANY-maze is too expensive and they score by hand.

## What we CAN replicate (software only)

Position-based measures from a top-down video:

| Test | Metrics | Extra requirement |
|---|---|---|
| Open field | distance, speed, center vs edge time, entries | — |
| Elevated plus maze / zero maze | open vs closed arm time, entries | head dips need the pose model |
| Y-maze / T-maze | entries, alternation | — |
| Light/dark box, place preference | zone times, transitions | — |
| Novel object recognition | time near each object | needs the nose point |
| Barnes maze / water maze | path, latency, distance to target | — |
| Forced swim / tail suspension | immobility time | needs a classifier |
| Manual-scoring player | equivalent to TakeNote | — |

## What we CANNOT replicate (needs hardware)

- Shock grids, pellet dispensers, lick sensors, optogenetic lasers, audio, TTL sync to electrophysiology.
- Operant conditioning (levers, nose-poke ports).
- Stimulus delivery in fear conditioning. We can still score freezing from video.
- Closed-loop real-time triggering.
- Beam-break rearing frames, thermal gradient ring, waterwheel, OPAD.

## Infrared

- IR only lets the camera see in the dark. It does not improve measurement accuracy, and it does not solve behaviour scoring.
- Lit tests: the lab's existing camera is enough.
- Dark-phase tests: an off-the-shelf night-vision camera ($30–100) plus an IR illuminator ($20–50), not a proprietary rig. The same software runs on the IR video.
- Train on grayscale with IR-style augmentation (see the ivc-cage-showcase repo).

## Model plan and confidence

| Stage | What | Metrics | Confidence |
|---|---|---|---|
| 1 | YOLO11 detection + ByteTrack, body center | distance, speed, zones, entries | **High** — existing stack; YOLO11 shown working on IR cage video |
| 2 | Rodent pose, fine-tuned (standard YOLO-Pose is human-only) | head dips, object exploration, head direction | **Medium** — needs labelled mouse frames |
| 3 | Temporal behaviour classifier on pose sequences | rearing, grooming, freezing, floating/immobility | **Low** out of the box — needs labelled data and tuning |

Stage 2 keypoints: nose, left ear, right ear, body center, tail base. Optional: neck, tail middle, tail tip. Arena corners are used for calibration.

### v1 scope

- **In:** stage 1 only, open field + plus maze, plus a validation report showing our numbers match the reference answers.
- **Out:** live cameras, multiple animals, home-cage 24/7.

## Validation data (verified)

- **Sturman et al. videos** (Zenodo, CC-BY 4.0): <https://zenodo.org/records/3608658>
  - 20 open-field, 24 plus-maze and 29 forced-swim videos.
  - 928×576, 25 fps, ~10 min each.
  - Open field 45×45 cm (calibrated on 42×42). Plus-maze arms 65.5×5.5 cm. Forced-swim beaker 20 cm.
- **Answer keys** (GPL-3.0): <https://github.com/ETHZ-INS/DLCAnalyzer>
  - Labels from 3 human raters, DLC tracking CSVs, zone files.
  - Example: OFT_3 distance ≈ 4008 cm, center time ≈ 79.8 s.
  - Caveat: check which video file matches which result file.
- **Paper:** <https://www.nature.com/articles/s41386-020-0776-y>
- **Kumar Lab open field:** ~16k labelled frames, includes black mice and 940 nm IR night video. <https://www.kumarlab.org/2019/02/12/single-mouse-tracking-annotated-dataset/>
- **IR and dark datasets:**
  - Luxem/VAME open field with IR from below.
  - DeepEthoProfile: 3.24M IR home-cage frames labelled with 8 behaviours.
  - MARS (CaltechDATA): IR home-cage video with pose keypoints.
- **Template for YOLO on IR mouse video:** <https://github.com/StalmokasGrazvydas/ivc-cage-showcase>

## Market (honest status)

- These tests appear in 10,000+ papers, while EthoVision is at ~2,000 sites. That points to a large unserved share, but this is an inference, not proven. Some of the gap is other tools, free DeepLabCut, or labs happy to score by hand.
- **Positioning:** don't sell "the same as your licence" to labs that already own one. Target labs with no licence, and the cases where paid tools fail (dark video, low contrast, complex behaviours, home-cage).
- **Armenian pilot candidates:** Orbeli Institute of Physiology, and the Mndzhoyan Institute of Fine Organic Chemistry (seizure, plus-maze and open-field studies). Armenia is a pilot ground, not the market.

## Open items

- [ ] Bottom-up count of reachable labs, by region.
- [ ] Specify one concrete IR camera + illuminator kit.
- [ ] **Next concrete step:** run stage 1 on the Sturman open-field videos and compare distance and center time with the DLCAnalyzer answers.
