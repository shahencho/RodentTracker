# Phase A: navigation shell and placeholders

*Subtask of [anymaze_parity_plan.md](anymaze_parity_plan.md). Started 2 November 2026.*

## Goal

Lay the UI out the way ANY-maze is laid out, so that a person who knows ANY-maze finds every section from its demo and can see at once whether it works or is planned. **No new analysis features**: everything that works today keeps working the same way.

## Scope

**In**
- New top tabs in ANY-maze order: **Protocol · Experiment · Tests · Results · Data**, then our own **Validation** and **Roadmap**.
- The Setup page becomes the **Protocol** page. Its left panel is grouped like the ANY-maze protocol sidebar: Apparatus, Tracking, Behaviour, Testing, Reports. The existing controls move into those groups unchanged.
- **Placeholders** for every planned feature, in the place ANY-maze users would look for it.
- **Tests** tab with real content: one row per video, with its status. Clicking a row opens that test.
- A **Roadmap** tab listing every planned feature by phase.
- A header switch, **Show planned features**, to hide all placeholders for a clean demo. The browser remembers the choice.

**Out**
- Any Phase B–D behaviour: experiments, animals, queues, group statistics, named zones.
- Changes to the server API or the analysis.

## Placeholder rules

1. **One list in the code.** Every planned feature is one entry in `app/static/planned.js`: id, where it goes (slot), name, what it will do, phase. The page, the Roadmap tab and the test all read this list.
2. **One look.** A grey row with the feature name and a coloured badge: **Phase B**, **Phase C**, **Phase D**, **Later** or **Not in v1**.
3. **Never looks broken.** A placeholder has no working buttons or inputs. Hovering shows what it will do; clicking it expands the same text. It never changes the setup or calls the server.
4. Planned test types (Y-maze, water maze, …) appear in the Test dropdown as **disabled** options marked "planned".

## Where each placeholder goes

| Tab › section | Placeholder | Phase |
|---|---|---|
| Protocol › top | Save and load the protocol | B |
| Protocol › Apparatus › Video source | Live camera | Not in v1 |
| Protocol › Apparatus › Test | Y-maze, T-maze, light-dark box, novel object, water maze, Barnes maze (disabled options) | Later |
| Protocol › Apparatus › Test | Round arenas and drawing tools (ellipse, line, multi-line) | Later |
| Protocol › Tracking | Head and tail tracking | Later |
| Protocol › Behaviour › Zones | Named zones from grid squares | D |
| Protocol › Behaviour › Zones | Hidden and investigation zones | Later |
| Protocol › Behaviour › Zones | Points | Later |
| Protocol › Behaviour | Freezing detection | Later |
| Protocol › Behaviour | Turning and heading | D |
| Protocol › Behaviour | Sequences | D |
| Protocol › Behaviour | Manual scoring with keys | D |
| Protocol › Testing | Test duration, stages and trials | B |
| Protocol › Testing | Procedures (end the test on an event) | Later |
| Protocol › Reports | Choose the measures to report | C |
| Experiment | Title and notes; animals and treatment groups; import animals from CSV | B |
| Tests | Run all tests in a queue | B |
| Tests | Live testing view and session log | Not in v1 |
| Results | Group results (mean, SD, SE by treatment); graph view; statistics (t-test, ANOVA + Dunnett); group track plots and heat maps; experiment PDF | C |
| Data | All tests in one table; CSV and Excel export | C |

The Experiment and Data tabs also show a greyed preview of their future layout (title, notes and an animals table; a results table with column headers), so the gap is visible at a glance.

## Files

| File | Change |
|---|---|
| `app/static/planned.js` | New: the list of planned features and phases |
| `app/static/index.html` | New tabs, grouped Protocol panel, placeholder slots, Experiment, Tests, Data and Roadmap pages |
| `app/static/app.js` | Render placeholders and the Roadmap, Tests list, show/hide switch, tab renamed to `protocol` |
| `app/static/style.css` | Group headings, placeholder rows, badges |
| `experiments/ui_check.py`, `ui_check_epm.py` | Tab name `setup` → `protocol` |
| `experiments/ui_check_phase_a.py` | New headless check for Phase A |

## Test

`experiments/ui_check_phase_a.py` (headless Chromium, server running) checks:

1. The 7 tabs exist in order, and each opens its page.
2. Every entry in the list appears exactly once: in its slot, or as a disabled Test option. Every slot on the page has at least one entry. The Roadmap lists every entry.
3. Placeholders contain no enabled controls, and clicking one changes nothing: the setup is unchanged and no request is sent.
4. The planned Test options are disabled.
5. The Tests tab lists every video from `/api/videos`. Clicking an analysed video opens its results; clicking an unanalysed one opens the Protocol page.
6. **Show planned features** off hides every placeholder, and the choice survives a reload.
7. No JavaScript errors. Screenshots of each tab go to `outputs/ui_check/phase_a_*.png`.

Regression: the existing `ui_check.py` (open field) and `ui_check_epm.py` (plus maze) must still pass.

## Done when

- All checks above pass.
- You have opened the UI, walked through the tabs and confirmed the layout. **Phase B starts only after that confirmation.**
