"""Prepare the demo UI with what we already computed, so it opens with results instead of a 5-10 minute run.

- YSMU: saves the setup of the first video from calib/ysmu_cage22.json (the second video then shows
  "copied from the saved setup of ..."), and copies cached detections from outputs/analysis/<video>/.
- Sturman test videos: setup from calib/sturman/<video>.json (42 x 42 cm) and detections cached by the validation.

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
