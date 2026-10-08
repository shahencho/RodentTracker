"""Call the SuperAnimal-TopViewMouse Faster R-CNN detector directly and print its raw top scores (run with .venv-dlc).

    .venv-dlc\\Scripts\\python scripts\\probe_sa_detector.py <clip.mp4> [n_frames]
"""
import sys

import cv2
import torch
import torchvision

CKPT = ".venv-dlc/Lib/site-packages/deeplabcut/modelzoo/checkpoints/superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt"
clip = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12

model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=2, box_score_thresh=0.0)
sd = {k.removeprefix("model."): v for k, v in torch.load(CKPT, map_location="cpu", weights_only=False)["model"].items()}
print(model.load_state_dict(sd, strict=False))
model.eval()

cap = cv2.VideoCapture(clip)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for k in range(n):
    idx = int(k * (total - 1) / max(n - 1, 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    img = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        out = model([img])[0]
    top = [(round(s, 3), [int(v) for v in b]) for s, b in zip(out["scores"][:3].tolist(), out["boxes"][:3].tolist())]
    print(f"frame {idx}: top scores {top}", flush=True)
