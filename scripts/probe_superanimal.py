"""Zero-shot test of DLC SuperAnimal-TopViewMouse on short probe clips (run with .venv-dlc).

    .venv-dlc\\Scripts\\python scripts\\probe_superanimal.py outputs\\probe_superanimal\\an1_120frames.mp4 [detector]
"""
import sys
import time

import deeplabcut

video = sys.argv[1]
detector = sys.argv[2] if len(sys.argv) > 2 else "fasterrcnn_resnet50_fpn_v2"
t0 = time.time()
deeplabcut.video_inference_superanimal(
    [video],
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name=detector,
    dest_folder=f"outputs/probe_superanimal/{detector}",
    max_individuals=1,
    bbox_threshold=0.1,
    pcutoff=0.1,
    device="cpu",
)
print(f"DONE {video} {detector} in {time.time() - t0:.0f}s", flush=True)
