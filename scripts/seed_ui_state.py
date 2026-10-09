"""Prepare the demo UI with what we already computed, so it opens with results instead of a 5-10 minute run.

- YSMU: saves the setup of the first video from calib/ysmu_cage22.json (the second video then shows
  "copied from the saved setup of ..."), and copies cached detections from outputs/analysis/<video>/.
- Sturman open-field test videos: setup from calib/sturman/<video>.json (42 x 42 cm), detections from the validation.
- Sturman plus-maze videos: 12 maze points from the DLC tracks, detections from validate_epm.py.

    .venv\\Scripts\\python scripts\\seed_ui_state.py
"""
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.server import state_dir  # noqa: E402
from src.analysis import Setup  # noqa: E402
from src.calibration import Calibration  # noqa: E402


def seed(video, calib_path, raw_src, save_setup=True):
    d = state_dir(video)
    if save_setup and calib_path:
        c = Calibration.load(calib_path)
        Setup(c.corners_px.tolist(), c.width_cm, c.height_cm).save(os.path.join(d, "setup.json"))
    if raw_src and os.path.exists(raw_src):
        shutil.copy(raw_src, os.path.join(d, "raw_track.csv"))
    print(f"{video}: setup {'saved' if save_setup and calib_path else '-'}, "
          f"detections {'copied' if raw_src and os.path.exists(raw_src) else 'missing'}")


ysmu = sorted(glob.glob("data/ysmu/*.mp4"))
for i, v in enumerate(ysmu):
    stem = os.path.splitext(os.path.basename(v))[0]
    seed(v.replace("\\", "/"), "calib/ysmu_cage22.json", f"outputs/analysis/{stem}/raw_track.csv", save_setup=(i == 0))

for raw in sorted(glob.glob("outputs/validation/mouse_det_v2/OFT_*_raw.csv")):
    vid = os.path.basename(raw)[:-len("_raw.csv")]
    seed(f"data/sturman/oft/{vid}.mp4", f"calib/sturman/{vid}.json", raw)

# Sturman EPM videos: the 12 maze points from the DLC tracks (12-corner mode), detections cached by validate_epm.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from dlcanalyzer_ref import epm_medians, read_dlc  # noqa: E402
from epm_quickcheck import dlc_file_for as epm_dlc_file  # noqa: E402

for raw in sorted(glob.glob("outputs/validation/mouse_det_v2_epm/EPM_*_raw.csv")):
    vid = os.path.basename(raw)[:-len("_raw.csv")]
    video = f"data/sturman/epm/{vid}.mp4"
    if not os.path.exists(video):
        continue
    points = {p: v.tolist() for p, v in epm_medians(read_dlc(epm_dlc_file(vid))).items()}
    Setup(test="epm", maze_mode="corners", maze_points_px=points).save(os.path.join(state_dir(video), "setup.json"))
    shutil.copy(raw, os.path.join(state_dir(video), "raw_track.csv"))
    print(f"{video}: plus-maze setup from DLC maze points, detections copied")
