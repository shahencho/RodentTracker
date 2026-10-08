# Open-source rodent detectors/trackers with released trained weights (GitHub + papers)

Scope: YOLO-family and other lightweight detectors for lab mouse/rat tracking that ship reusable weights. DeepLabCut SuperAnimal, SLEAP and MMPose are out of scope here (covered by another researcher); cross-link only: Kumar Lab's HRNet pose and DAMM are adjacent alternatives. Research date: 2026-10-08. The GitHub REST API was queried directly for repo metadata, file trees and releases. Items marked (UNVERIFIED) come from prior knowledge and were not checked this session.

## Q1. StalmokasGrazvydas/ivc-cage-showcase: model, weights, training data, licence

### Takeaway
This is a portfolio showcase, not a code or weights release. It describes a YOLOv8n mouse detector fine-tuned on grayscale IR home-cage frames, but the repo contains only a README and images. There are no weights and no licence file, so it cannot be reused.

### Cited Findings
- The model is YOLOv8n (chosen over YOLO11n). It is paired with a Hungarian-assignment tracker and a Random Forest behaviour-state classifier. — [README](https://github.com/StalmokasGrazvydas/ivc-cage-showcase)
- Training data: a Roboflow dataset (5,314 train / 368 val / 262 test images), then domain adaptation on 141 real-cage frames from client footage. All images were converted to grayscale to match the infrared camera. Augmentations were brightness/contrast drift, gamma, CLAHE, sensor noise, motion blur and JPEG artefacts. — [README](https://github.com/StalmokasGrazvydas/ivc-cage-showcase)
- Real-cage test set (only 20 images): mAP50 0.907, mAP50-95 0.459, P 0.944, R 0.822. The authors call these "indicative, not precise". — [README](https://github.com/StalmokasGrazvydas/ivc-cage-showcase)
- CPU speed: ONNX Runtime on CPU (Colab) ran at about 7.5 FPS; a T4 GPU ran at about 64 FPS. — [README](https://github.com/StalmokasGrazvydas/ivc-cage-showcase)
- The GitHub API file tree holds only `.gitignore`, `README.md` and `assets/` images (prototype photos and result plots). There are no .pt/.onnx files and no releases. The API reports `license: null`. The README says the code lives in a private team repo under MIT. — [GitHub API tree](https://api.github.com/repos/StalmokasGrazvydas/ivc-cage-showcase/git/trees/main?recursive=1)

### Inferences
- Even if obtainable, a model trained on IR home-cage footage (side or oblique cage view, bedding background) is a domain mismatch for a top-down white open field. Its augmentation recipe (grayscale, CLAHE, noise, blur) is a good template for our own fine-tune.
- The underlying Roboflow dataset is unnamed. It may be findable on Roboflow Universe as a training source.

### Gaps
- Which Roboflow dataset was used is not stated. The private repo's weights are not accessible.

## Q2. Kumar Lab (JAX): Geuther et al. 2019 tracking network and later pose/gait models

### Takeaway
The 2019 single-mouse open-field tracker does have released pretrained models. It was trained on a wide range of coat colours, including albino, with white-floor arenas and IR footage, at 480x480 grayscale top-down. That makes it the closest published domain match to our video. However, it is TensorFlow 1.x (Python 2.7) code, which is a deployment burden. The later HRNet pose model is PyTorch and non-commercial only.

### Cited Findings
- Paper: Geuther et al. 2019, "Robust mouse tracking in complex environments using neural networks", Communications Biology (PMC6440983). The authors provide "a labeling interface, labeled training data, tuned hyperparameters, and a pretrained network". — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s42003-019-0362-1&resultType=core&format=json)
- Coat colours covered include black, agouti, albino, gray, brown, nude and piebald. Lighting was visible light (about 600 lux; 300–1000 lux variants) plus 940 nm IR LED for night recording. Arenas were a white PVC floor with grey walls, and the clear/reflective KOMP2 commercial boxes. Images were 480x480 (captured at 640x480, 30 fps). — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6440983/)
- Accuracy: the encoder–decoder segmentation network had a validation centre error of about 0.9 px and mean IoU 0.828 (vs 0.743 for the regression network). Ctrax degraded badly with albino mice and reflective surfaces. The network correlated with beam-break at R = 96.9%. Speed was about 200 fps on a P100 and about 125 fps on a Titan XP; no CPU figures were given. — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6440983/)
- Pretrained models are on a live kumarlab.org page with downloads `ftp://ftp.jax.org/kumarlab/2019_Tracking/Pretrained_Models.tar` and Zenodo record 5806397. Five models are listed: AllCoatColor_RegularOFA, 24Hr_OFA, KOMP_OFA, FullModel and NoHardFrames_FullModel. The page gives no file size, format or licence. — [kumarlab.org models page](https://www.kumarlab.org/2019/02/12/pre-trained-single-mouse-tracking-neural-network-models/); the annotated dataset is on a [separate page](https://www.kumarlab.org/2019/02/12/single-mouse-tracking-annotated-dataset/)
- Code: [KumarLabJax/MouseTracking](https://github.com/KumarLabJax/MouseTracking), MIT licence, last push 2019-02-14. It needs TensorFlow 1.0.1–1.4 and Python 2.7.12. There are three architectures (segmentation-based ellipse regression, direct ellipse regression, binned XY), all "designed for 480x480 monochromatic images" for single mice in open fields. — [README](https://github.com/KumarLabJax/MouseTracking); licence and last push confirmed via the GitHub API
- Later pose/gait work: Sheppard et al., "Stride-level analysis of mouse open field behavior using deep learning-based pose estimation". Zenodo 6380163 contains `pose-est-model.pth` (114.8 MB, PyTorch HRNet), training annotations (427.6 MB h5) and a 3.6 GB Singularity image, under a non-commercial licence. Code is in the `deep-hrnet-mouse` and `gaitanalysis` repos. — [Zenodo 6380163](https://zenodo.org/records/6380163)

### Inferences
- Of everything found, the Kumar 2019 models (especially AllCoatColor_RegularOFA or FullModel) best match our case: top-down, monochrome, albino-on-white-floor, IR-capable, single mouse. Our 640x480 frames would need cropping or resizing to 480x480.
- The format is almost certainly TF1 checkpoints, given the TF1 code (not verified by download). Running it on CPU means either a legacy TF1 environment or converting the graph (e.g. tf2onnx). The output is an ellipse (centre, axes, angle), not a box. That suits centroid tracking well.
- The HRNet pose model (12 keypoints, UNVERIFIED count) is heavy (115 MB, HRNet-W32-class). It would be slow but possible on CPU, and its non-commercial licence limits use.
- Rearing and tilted-camera frames were probably underrepresented in the training data, which used a top-down fixed camera.

### Gaps
- Exact file format, size and licence of Pretrained_Models.tar and Zenodo 5806397 were not verified (not downloaded or fetched). Whether the JAX FTP is still reachable was not tested.
- No CPU inference benchmarks were published.

## Q3. Other GitHub repos and papers with downloadable rodent weights

### Takeaway
Few YOLO rodent projects actually publish weights. The usable ones found are: S4H0ST RatTracker (YOLOv8s-pose, white rats in a white arena, weights only inside a release exe bundle), OpenFieldAI (YOLO .pt models in a 2.1 GB SourceForge zip, GPLv3), and DAMM (Mask R-CNN, zero-shot mouse detector, CC BY-NC-ND). YOLO-MousePose has no mouse weights.

### Cited Findings
**S4H0ST/Computer-Vision-Mice-Tracking ("RatTracker Pose")**
- YOLOv8s-Pose with 3 keypoints (snout, spine, tail). It is trained on 1,101 images (augmented from 248 originals) of "white rats in a hole-board arena" in a single "white arena" filmed overhead under constant lighting. Reported results: mAP50 0.82, recall 0.887, 92.1% detection rate on a test video. Licence is GPL-3.0. — [README](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking)
- The README says the weights are at `models/yolo_ratas.pt`. The GitHub API tree shows `models/` holds only `.gitkeep`, so the .pt is not in the repo. Release v1.0 (2026-09-13) ships `RatTracker_lab.zip` (349.7 MB), a standalone executable that the README says bundles the model. — [GitHub API releases](https://api.github.com/repos/S4H0ST/Computer-Vision-Mice-Tracking/releases); [direct zip](https://github.com/S4H0ST/Computer-Vision-Mice-Tracking/releases/download/v1.0/RatTracker_lab.zip)

**OpenFieldAI** (J Neurosci Methods 2025, S0165027025003115)
- A Python GUI for the open field test that uses YOLO. It ships pretrained "OFAI 160, 320, 640" models (.pt) and supports custom YOLOv8 training. GPLv3, Windows. Download is OFAInew.zip (2.1 GB), last updated 2026-04-23. — [SourceForge](https://sourceforge.net/projects/openfieldai/); [paper abstract](https://www.sciencedirect.com/science/article/abs/pii/S0165027025003115) (full text returned 403); [search summary](https://www.sourceforge.net/p/openfieldai/code)
- The model's YOLO version and its training data (species, coat colour, arena) were not stated on the pages accessible this session.

**DAMM: Detect-Any-Mouse-Model** (Kaul et al., Sci Rep 2024)
- A Mask R-CNN (LVIS-pretrained) trained on 2,200 annotated images drawn from about 12,500 videos. Views are top and side. Coat colours are black, agouti, white and others. Settings include home cages, mazes and arenas, under white and red light, in RGB and grayscale. — [PMC11393305](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305/)
- Zero-shot AP75 on unseen setups: 85.6–100% for simple and home-cage scenes, 58.1% for a complex low-light apparatus, 95.9–99.5% on public datasets, and 26.9% on rats. Speed is 11 FPS on a V100 and 18 FPS on an L40. Code, weights and data are released under CC BY-NC-ND 4.0. — [PMC11393305](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305/)
- The repo serves weights via Dropbox links and now pairs detection with SAM 2 for video segmentation. It was tested on Linux with an NVIDIA GPU, and CPU use is not discussed. — [backprop64/DAMM](https://github.com/backprop64/DAMM); [project page](https://web.eecs.umich.edu/gkaul/DAMM)

**YOLO-MousePose** (bujihao; IEEE TIM vol. 74, 2025)
- An improved YOLO-Pose for top-down open-field mouse pose, trained on the MPD-OFT dataset. The dataset is "will be open sourced after the paper is reviewed". — [README](https://github.com/bujihao/YOLO-MousePose); [IEEE](https://ieeexplore.ieee.org/document/10929680)
- The repo's `weights/` holds only `yolobackbone.pth` (12.7 MB, a generic backbone) and a `download_weights.sh` that fetches stock yolov5s/m/l/x COCO weights. There are no releases and no licence (API `license: null`). — [GitHub API](https://api.github.com/repos/bujihao/YOLO-MousePose/contents/weights)

**Other leads found but not verified this session**
- "A YOLO based software for Automated Detection and Analysis of Rodent Behaviour in the Open Field Arena" (2021) and Cosevare, a YOLOv3 open-field mouse tracker. Weight availability is unknown. — [ResearchGate](https://www.researchgate.net/publication/351609553_A_YOLO_based_software_for_Automated_Detection_and_Analysis_of_Rodent_Behaviour_in_the_Open_Field_Arena)
- A Hugging Face model "kaopanboonyuen/DeepRodent" appeared in search; its contents were not checked. — [HF](https://huggingface.co/kaopanboonyuen/DeepRodent)
- vailá "animal open field" module (docs page appeared in search; not checked). — [docs](https://vaila.readthedocs.io/en/latest/vaila_buttons/animal-open-field/)
- LabGym ([umyelab/LabGym](https://github.com/umyelab/LabGym), GPL-3.0, active as of 2026-10-01 per the API) uses Detectron2 "Detectors" that users train themselves. No official pretrained mouse detector download was found on PyPI or in search. Its "Subtract Background" mode is classical. — [PyPI](https://pypi.org/project/LabGym/)
- Not detectors, or no pretrained weights (from prior knowledge, UNVERIFIED this session):
  - ezTrack: classical background subtraction, no NN weights.
  - idtracker.ai and TRex: segmentation by thresholding plus identity CNNs trained per video; no reusable detector.
  - DeepEthogram: a behaviour classifier, not a detector.
  - AlphaTracker: multi-mouse pose (YOLOv3 + AlphaPose); demo weights were reportedly on Google Drive.
  - MouseTracker: unclear.

### Inferences
- **Best bets for a small white mouse in a white arena, CPU-only:**
  1. **Kumar Lab 2019 FullModel / AllCoatColor**: explicitly trained on albino mice on a white floor, monochrome, top-down, IR. It has an MIT code licence and is the best domain fit. Its cost is TF1 legacy tooling.
  2. **S4H0ST RatTracker `yolo_ratas.pt`**: the only YOLO .pt found that was trained on white animals in a white arena. It is a YOLOv8s-pose that runs on CPU via ultralytics or ONNX. Two caveats: the animals are rats, not mice (larger, at unknown pixel size), and the .pt must be extracted from the 350 MB PyInstaller exe (feasible with pyinstxtractor, but GPL-3.0 applies). The training set is small (248 originals from one setup), so generalisation is doubtful.
  3. **DAMM**: the most general zero-shot mouse detector (white mice and grayscale both in its training data). It is heavy (Mask R-CNN, about 11 FPS on a V100, so probably about 1 FPS or less on CPU), but it is usable offline to auto-label frames for training our own YOLO11n. Its NC-ND licence restricts redistribution of derivatives.
  4. **OpenFieldAI OFAI models**: drop-in YOLO .pt files, but their training domain is unknown. Worth a quick test given the 2.1 GB download.
- A practical route: use DAMM or the Kumar model as a "teacher" to pseudo-label a few hundred frames of our own videos, then fine-tune YOLO11n/YOLOv8n (exported to ONNX) for CPU speed. This mirrors the IVC showcase recipe of grayscale plus CLAHE/noise augmentation.

### Gaps
- None of the weights were downloaded and smoke-tested on our 640x480 grayscale test video.
- OpenFieldAI training data and YOLO version, Cosevare weight availability, DeepRodent HF contents, and the AlphaTracker and LabGym pretrained-model status all remain unverified.
- No source reported CPU FPS for the Kumar or DAMM models.
