# ANY-maze parity: what we lack and the plan

*Started 2 November 2026. Source: the Stoelting video "An introduction to ANY-maze" (8 min, no audio), reviewed frame by frame.*

## Goal

People who use ANY-maze should open RodentTracker and recognise the same workflow, even where a feature is not built yet. Every ANY-maze feature we know of gets one of these places:

- **Have:** built and working.
- **MVP+:** to build next, because a demo to ANY-maze users looks incomplete without it.
- **Later:** planned, but needs research (pose, behaviour classifiers) or is outside v1. It appears in the UI as a visible, disabled **placeholder**.
- **Out:** not planned (hardware, live recording). It gets a placeholder only where ANY-maze users would look for it.

## Where the MVP line sits

The v1 scope (handoff) is: upload a video, pick the test, mark the arena and zones, get metrics, a CSV and a PDF, for the open field and the plus maze, with validation against a reference. That is done for **one video at a time**.

ANY-maze's real unit of work is the **experiment**: many animals, treatment groups, trials, then statistics across groups. Half the demo video shows that, and we have none of it. **This is the main thing to add before a demo to ANY-maze users**, so it moves from "later" into MVP+.

## Feature map

| ANY-maze (as shown in the video) | Status | Notes |
|---|---|---|
| **Protocol** | | |
| Video source: video file | Have | Upload .mp4 |
| Video source: live camera | Out → placeholder | Live recording is out of v1 |
| Operating modes: I/O, freezing, forced swim / tail suspension | Out / Later → placeholder | FST/TST are a possible later test type |
| Apparatus: draw rectangle, ruler line for scale | Have | 4 floor corners and the real size; also corrects for tilt |
| Apparatus: ellipse, line, multi-line tools (round arenas, water maze, Barnes) | Later → placeholder | Needs a circular calibration and zone template |
| Apparatus templates beyond OF and EPM (Y-maze, T-maze, light-dark, novel object, water maze) | Later → placeholder | Each is built like the EPM: template, zones, reference |
| Animals: species, lighter/darker than background | Have (implicitly) | The detector learns this; each new lab needs one CPU adaptation |
| Head and tail tracking | Later (stage 2) → placeholder | Pose model |
| Zones from clicked grid cells, any number, named | **MVP+** | Today: centre (with toggled cells), wall, fixed 20% corners only |
| Hidden and investigation zones, zones whose position varies per test | Later | Investigation zones need head points |
| Points | Later | Distance and heading to a point; heading needs the head |
| Immobility detection | Have | DLCAnalyzer rule |
| Freezing detection | Later (stage 3) → placeholder | Can start with a pixel-change rule, without pose |
| Orientation and movement (turning, heading) | Later → placeholder | Turning from the track is feasible; heading needs the head |
| Sequences (event chains, e.g. zone A then B) | Later → placeholder | Builds on named zones |
| Keys: manual scoring with the keyboard during playback | **MVP+ (small)** | Cheap and useful for rearing or grooming until stage 3 |
| Virtual switches, I/O | Out | Hardware |
| Stages: test duration, trials, testing order | **MVP+** | Test duration from the protocol sets the end time automatically |
| Procedures (end the test on an event, e.g. island reached) | Later | Comes with the water maze |
| Save and load the protocol | **MVP+** | Today the setup is stored per video |
| **Experiment** | | |
| Title, notes | **MVP+** | |
| Animals with status and treatment group | **MVP+** | |
| Import animals | MVP+ (CSV) | |
| Experiment protection and encryption | Out | |
| **Tests** | | |
| Test list: animal × stage × trial, with status | **MVP+** | One video per test |
| Run a queue of tests | **MVP+** | Background queue, about real time per video on a CPU |
| Live tracking view, session log, live zone counts | Out → placeholder | We analyse after recording; the playback overlay already shows the path |
| **Results** | | |
| Choose the measures to report (checklist) | **MVP+** | Today the set is fixed |
| Extra measures: first zone entered, zones visited, speed and distance per zone, distance to a zone | MVP+ (part) / Later | Track-only measures are cheap |
| Group by treatment, then by other fields; filter by stage and trial; exclude retired animals | **MVP+** | |
| Text view: N, mean, SD, SE, raw data | **MVP+** | |
| Graph view: bars with error bars | **MVP+** | |
| Statistical view: one-way ANOVA + Dunnett's test against the control | **MVP+** | The ANY-maze example gives a free answer key, see below |
| Track plots and heat maps per test and per group | Have per test; group is MVP+ | |
| **Data** | | |
| Spreadsheet of all tests × measures; save and export | **MVP+** | CSV and Excel |

## Phases

Effort is in working days for one developer. Complexity is how much is unknown, not how much code.

### Phase A: navigation shell and placeholders (1–2 days, low complexity)

Detailed subtask: [phase_a_navigation_plan.md](phase_a_navigation_plan.md).

Make the UI look like ANY-maze's structure now, so what's missing is visible.

- Top tabs: **Protocol · Experiment · Tests · Results · Data**, plus **Validation** (ours, kept).
  - Today's Setup page becomes **Protocol** (test type, arena, zones, timing) plus a per-test step for marking the corners.
  - Today's Results page becomes the **single-test result**, opened from the Tests list.
- Protocol sidebar like ANY-maze's: Apparatus, Tracking, Behaviour, Testing, Analysis, Reports. Built items work; the others are disabled with a **"Planned"** badge and a one-line tooltip saying what it will do and in which phase.
- Placeholders to add: live camera source; ellipse, line and multi-line tools; more apparatus types; head and tail tracking; freezing; orientation; sequences; investigation zones; points; procedures; the statistical view (until Phase C).
- **Rules for placeholders:** never clickable in a way that looks broken; one consistent badge; a single list in the code (`PLANNED_FEATURES`) so the UI, this document and a "Roadmap" page come from one source.

**Done when:** a person who knows ANY-maze can find every section from the video and see whether it works or is planned.

### Phase B: experiment layer (6–8 days, medium complexity)

- **Data model**, stored as JSON per experiment under `data/experiments/<id>/`:
  - `Protocol`: test type, arena size, zones, test duration, stages and trials.
  - `Animal`: id, treatment, status (normal or retired), optional fields such as sex.
  - `Test`: animal, stage, trial, video, arena corners, start time, status, result.
- **Protocol** is saved once and reused by every test, and can be exported and imported as a file.
- **Per-test step:** the camera can move between videos, so each test still needs its corners. Default to the previous test's corners, show them over the first frame, and let the user confirm or drag. The existing hand detection sets the start time; the end is start + test duration.
- **Run queue:** a background job runs the tests one by one, with progress, cancel, and resume after a restart. 12 ten-minute videos take about 2.5 hours on a CPU, so the queue must survive closing the browser.
- **Migration:** today's per-video setups become one-test experiments, so nothing is lost.
- **Main risk:** the corner step for many videos is tedious. Mitigation: reuse corners, and later detect the arena automatically.

**Done when:** a 12-animal, 2-group experiment can be set up, queued and run without touching each video's settings except confirming corners.

### Phase C: group results and Data tab (5–7 days, low–medium complexity)

- **Data tab:** one row per test and one column per measure, plus animal and treatment fields; CSV and Excel export.
- **Measure checklist** in the protocol (the ANY-maze "Test details report"), saved with the protocol.
- **Results tab:**
  - group by treatment, then up to two more fields; filter by stage and trial; exclude retired animals;
  - **Text:** N, mean, SD, SE, raw values;
  - **Graph:** bars with SD or SE;
  - **Statistical:** two groups use a t-test; more groups use a one-way ANOVA + Dunnett's test against a chosen control. Needs `scipy` (not installed yet);
  - group track plots and heat maps.
- **Free answer key for the statistics.** The ANY-maze water-maze example in the video shows the raw values and its results. Our ANOVA and Dunnett must reproduce them:
  - Saline: 19.0, 8.5, 11.3, 3.7, 11.4, 2.4
  - Compound X 1.0 mg/kg: 3.0, 30.7, 24.3, 10.3, 25.2
  - Compound X 5.0 mg/kg: 43.5, 35.7, 51.9, 26.2, 10.0
  - Compound X 10.0 mg/kg: 118.5, 110.3, 85.9, 53.1, 51.0
  - Expected: F(3,17) = 17.0806, p < 0.0001; Dunnett q = 0.8406 (p = 0.4122), 2.1724 (p = 0.0803), 6.7110 (p < 0.0001).
  - ANY-maze uses the **step-down** Dunnett (Dunnett and Tamhane 1991). `scipy.stats.dunnett` is single-step, so the p-values may differ slightly. Check, and port the step-down version if needed.
- **PDF:** an experiment report (groups, graphs, statistics) next to the per-test report.

**Done when:** the numbers above are reproduced, and an experiment's results can be exported in the same layout as ANY-maze.

### Phase D: zones, measures and manual scoring (5–7 days, medium complexity)

- **Named zones from grid cells:** any number of zones, each a set of clicked cells with a name and colour. The raster masks in `src/zones.py` already support this; the main work is the UI. Centre, wall and corners become presets.
- **Track-only measures:** first zone entered, list of zones visited, distance and speed inside each zone, distance to a zone and to its border, turning (rotations from the path direction). Each must be checked against a definition; DLCAnalyzer or the ANY-maze help pages are the references.
- **Keys:** keyboard scoring during playback (start/stop of a behaviour or single events), saved with the test and reported like zones.
- **Sequences:** simple chains of zone entries (e.g. Y-maze alternation). Leads into the Y-maze test.

**Done when:** a user can define "NW corner", "NE corner" and so on, see each zone in the results, and score rearing by hand.

### Later: research and new tests (each is a project, not a phase)

| Item | Complexity | Depends on |
|---|---|---|
| Round arenas (ellipse tool and calibration), water maze, Barnes maze | Medium | Phase D zones; a reference dataset |
| Y-maze, T-maze, light-dark box, novel object | Medium per test | Like the EPM: template, zones, reference data |
| Freezing (pixel-change rule first) | Medium | A fear-conditioning video with a reference score |
| Head and tail tracking (stage 2) | High | Pose model; unlocks heading, investigation zones, head dips |
| Rearing, grooming classifier (stage 3) | High | Labelled behaviour data |
| Live camera, real-time tracking, I/O | High; out of v1 | A product decision |

## Suggested order

1. **Phase A now.** It is cheap and makes every gap visible in the demo.
2. **Phase B, then C.** Together they give the experiment workflow, which is the biggest gap, and C has a ready answer key.
3. **Phase D.** Zones and manual scoring.
4. Resume the **EPM validation** ([epm_plan.md](epm_plan.md)) between phases. It is independent of this plan.
5. "Later" items are chosen by what YSMU and the pilot labs actually record (still open, project status section 6).

About 3–4 weeks of work for A–D.

## Open questions

- Does YSMU need groups and statistics in the tool, or do they export to SPSS or Prism? This decides how far Phase C goes.
- Does one experiment always use one camera setup? If so, the corners can be set once per experiment.
- What statistics do the pilot labs report? (t-test, ANOVA, Dunnett, Tukey, two-way ANOVA.)
