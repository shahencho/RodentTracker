"""Tune the YOLOE visual prompt: one example box vs. several, prompt embedded once.

Several examples are passed as one reference image: arena crops of different frames tiled 2x2,
so the mouse keeps its native size, each with its own box (same class -> embeddings averaged).
Usage: .venv\\Scripts\\python experiments\\tune_yoloe.py data\\ysmu\\cntrol60_cage22_an1_OF1.mp4
"""
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

sys.path.insert(0, os.path.dirname(__file__))
from probe_common import best_box, build_reference, pick_examples, reference_positions, sample_frames, score  # noqa: E402

video = sys.argv[1]
out = os.path.join("outputs", "probe", os.path.splitext(os.path.basename(video))[0])
frames = sample_frames(video)
refs = reference_positions(frames)


for n_ex in [1, 4, 6]:
    examples = pick_examples(refs, n_ex)
    ref_img, boxes = build_reference(frames, refs, examples)
    vis = ref_img.copy()
    for b in boxes.astype(int):
        cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(out, f"yoloe_prompt_{n_ex}ex.jpg"), vis)

    m = YOLOE("yoloe-11l-seg.pt")
    # embed the prompt once; afterwards plain predict() reuses it
    m.predict(ref_img, refer_image=ref_img, visual_prompts=dict(bboxes=boxes, cls=np.zeros(len(boxes), int)),
              predictor=YOLOEVPSegPredictor, verbose=False)
    for conf in [0.05, 0.01]:
        t = time.perf_counter()
        preds = [(b[:3] if b else None) for b in (best_box(m.predict(f, conf=conf, verbose=False)[0]) for _, f in frames)]
        ms = (time.perf_counter() - t) / len(frames) * 1000
        s = score(f"yoloe_l_{n_ex}ex_conf{conf}", frames, refs, preds, out)
        print(f"   examples at frames {[frames[j][0] for j in examples]}, {ms:.0f} ms/frame", flush=True)
