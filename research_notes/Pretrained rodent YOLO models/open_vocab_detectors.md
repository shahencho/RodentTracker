# Open-vocabulary / promptable foundation detectors and segmenters for zero-shot lab-mouse detection (CPU feasibility)

Context: top-down open-field / EPM video, 640x480 grayscale, 19 fps, ~30 px white mouse on white arena, tilted camera, wall-hugging/rearing. Stock COCO YOLO11 and YOLO-World text prompts ("mouse", "rat", "rodent") found the mouse in 0-1 of 120 frames. Research date: 2026-10-08. Items marked (UNVERIFIED) come from background knowledge and were not re-fetched this session.

## Q1. Per-model facts: weights, licence, Ultralytics integration, CPU speed, small-object performance

### Takeaway
Only YOLOE (and YOLO-World) are fast enough on CPU for frame-by-frame use. SAM 2/2.1 and SAM 3 work as trackers or labelers on CPU, not as real-time detectors. Grounding DINO, OWLv2 and Florence-2 run at roughly one or more seconds per frame on CPU, so they suit offline labelling of a few hundred frames. T-Rex2 and Grounding DINO 1.5/1.6 are cloud-API only. The best-matched pretrained option is not a general open-vocabulary model at all: DAMM (Detect-Any-Mouse-Model) is a Mask R-CNN trained specifically on diverse mouse setups, but its licence is non-commercial.

### Cited Findings
**YOLOE (Ultralytics; YOLOE-v8, YOLOE-11, YOLOE-26)**
- Supports text prompts, visual prompts (bboxes plus class IDs on a reference image via `YOLOEVPSegPredictor`), and prompt-free `*-seg-pf.pt` checkpoints with a built-in vocabulary of 4,585 names. Visual-prompt results come out labelled "object0", "object1", so you map the names yourself. — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- YOLOE-26 on LVIS minival at 640: 26x gets 40.6 mAP with text prompts and 38.5 with visual prompts; 26s gets 30.8 / 28.6 at about 22 GFLOPs. — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- The docs say "the n and s scales run on edge GPUs such as Jetson or on CPU at reduced resolution". Text prompting downloads a ~254 MB text encoder on first use. Exported models freeze the classes. "Zero-shot accuracy is well below a model trained on your classes." — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- With a video source, the first frame is used as `refer_image` by default. You can also pass any frame explicitly as `refer_image`. — [Ultralytics docs (KO mirror)](https://docs.ultralytics.com/ko/models/yoloe) / [Ultralytics YOLOE blog](https://www.ultralytics.com/blog/what-is-yoloe-taking-computer-vision-models-further)
- Paper: SAVPE visual-prompt encoder. Visual prompts "are effective for less frequent objects that text prompts often struggle to accurately describe". YOLOE-v8-S beats YOLO-Worldv2-S by 3.5 AP on LVIS at 1.4x the inference speed. Evaluation is on LVIS only; there is no animal or lab-rodent evaluation. — [YOLOE arXiv 2503.07465](https://arxiv.org/pdf/2503.07465)
- Licence: Ultralytics code and weights are AGPL-3.0, with a paid Enterprise licence for closed commercial use. (UNVERIFIED this session; the YOLOE docs page only says "check their repository".)

**YOLO-World v2**
- The Ultralytics YOLO-World integration supports text prompts only. Visual-prompt support arrived in YOLOE, its successor; YOLO-World "lacked features like instance segmentation and visual prompt support". — [Ultralytics blog](https://www.ultralytics.com/blog/what-is-yoloe-taking-computer-vision-models-further)
- Inference: you cannot fix the failed test by giving YOLO-World a visual prompt inside Ultralytics; switch to YOLOE instead. The original AILab-CVC YOLO-World repo had an image-prompt demo variant (UNVERIFIED this session, GPL-3.0 licence UNVERIFIED).

**Grounding DINO / 1.5 / 1.6**
- Grounding DINO gets 52.5 AP zero-shot on COCO. OWLv2 gets roughly 30-40 AP zero-shot on COCO versus 55.4 AP for Grounding DINO 1.6 Pro. — [Forasoft overview (secondary)](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr)
- Grounding DINO 1.5/1.6 and T-Rex2 come from IDEA Research and are served through a cloud API (deepdataspace). Grounding DINO 1.5 was announced 2024-05-17. — [IDEA-Research/T-Rex GitHub](https://github.com/IDEA-Research/T-Rex), [Grounding DINO 1.5 API repo](https://github.com/idea-research/grounding-dino-1.5-api)
- Original Grounding DINO (Swin-T/B) is Apache-2.0 with open weights, also available in HF transformers. On CPU it takes seconds per image. (UNVERIFIED this session; I found no published CPU benchmark.)
- Open-vocabulary detectors "exhibit dramatically degraded performance compared to their reported natural-image benchmarks" when moved to a new domain (aerial study). — [arXiv 2601.22164](https://arxiv.org/html/2601.22164v1)

**OWL-ViT / OWLv2**
- HF transformers supports one-shot detection guided by an image (`Owlv2ForObjectDetection.image_guided_detection`, processor `query_images=`). Query images override text prompts and the model matches image to image. Post-processing defaults to NMS 0.3. — [HF OWLv2 docs](https://huggingface.co/docs/transformers/model_doc/owlv2)
- The OWLv2 processor resizes input to 960x960 by default. A 640x480 frame is therefore upsampled, which costs compute on CPU (ViT-B/16 at 960 gives 3,600 patches). — [HF OWLv2 docs](https://huggingface.co/docs/transformers/model_doc/owlv2)
- OWL-ST self-training raises LVIS-rare AP from 31.2% to 44.6% (L/14). — [HF OWLv2 docs / paper abstract](https://huggingface.co/docs/transformers/model_doc/owlv2)
- Licence: Apache-2.0 (Google big_vision). (UNVERIFIED this session.)

**Florence-2**
- MIT licence. Base model has 0.23B parameters, large has 0.77B. Supports OD, open-vocabulary detection and phrase grounding. Roboflow and OpenVINO have CPU deployment guides, but "CPU inference may be very slow… for larger models". — [Roboflow Florence-2](https://roboflow.com/model/florence-2), [Roboflow Intel CPU blog](https://blog.roboflow.com/deploy-florence-2-intel-emerald-rapids/), [OpenVINO notebook](https://docs.openvino.ai/2024/notebooks/florence2-with-output.html)
- Florence-2 has no visual-exemplar prompt; it takes text or a region only.

**SAM 2 / SAM 2.1 (Ultralytics and Meta)**
- Ultralytics `SAM2VideoPredictor` tracks from a point or box prompt, e.g. `predictor(source='test.mp4', points=[920, 470], labels=[1])`. Sizes range from 78.1 MB (tiny) to 162 MB (base). — [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2/)
- Ultralytics CPU benchmark (Apple M4 Air): SAM2-t takes 23,430 ms per image and SAM2-b 28,867 ms, versus YOLO11n-seg at 24.3 ms. This benchmark is probably segment-everything mode; a single prompted object is cheaper, but it still gives the order of magnitude. — [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2/)
- `auto_annotate()` combines a YOLO detector with SAM 2 to generate segmentation labels. — [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2/)
- SAM 2 paper: streaming-memory transformer trained on SA-V (50.9K videos, 642.6K masklets). It needs 3x fewer interactions than prior work. — [arXiv 2408.00714](https://arxiv.org/abs/2408.00714)
- Licence: Apache-2.0 for code and checkpoints. (UNVERIFIED this session.)

**SAM 3 / 3.1**
- Released 2025-11-19, with 473.6M parameters and a 3.45 GB checkpoint. Weights are gated on Hugging Face (access request required). It does Promptable Concept Segmentation from text, image exemplars (boxes) or both, and video tracking. Ultralytics classes are `SAM3SemanticPredictor`, `SAM3VideoPredictor` and `SAM3VideoSemanticPredictor`. SAM 3.1 (2026-03-27) adds Object Multiplex. — [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3/)
- The Ultralytics page reports 2,921 ms per image on an RTX PRO 6000 GPU in its comparison; CPU would be far slower. — [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3/)
- Licence: Meta "SAM License", custom, not Apache. (UNVERIFIED; the Ultralytics page only says "open-source license".)

**T-Rex2**
- Takes text, visual box/point prompts, or both (interactive and generic visual-prompt workflows). — [arXiv 2403.14610](https://arxiv.org/abs/2403.14610v1)
- **API-only; no open weights.** The repo holds API code under "IDEA License 1.0", and you need an API key from the deepdataspace cloud. Free API access exists for education and research by request form. IDEA's newer Rex-Omni (2025-10-15) is described as "fully open-sourced". — [IDEA-Research/T-Rex GitHub](https://github.com/IDEA-Research/T-Rex)

**DAMM: Detect-Any-Mouse-Model (a domain-specific alternative)**
- Mask R-CNN trained on AER-LG: 2,200 annotated images drawn from 12,500 videos with varied views, coat colours, lighting and setups. — [DAMM, Sci Rep 2024 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305)
- Zero-shot on unseen public setups: 95.9-99.5% Mask AP75. Exceptions: a fisheye rat operant chamber scored 26.9%, and enriched multi-mouse cages went from 56.4% zero-shot to 84.7% with 20-shot fine-tuning. — [DAMM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305)
- Speed is 11 FPS on a V100 GPU (CPU not reported). Code, weights, data and a Colab notebook are at github.com/backprop64/DAMM. It uses SORT for tracking and a SAM-based annotation GUI. **Licence is CC BY-NC-ND 4.0 (non-commercial).** — [DAMM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305), [DAMM GitHub](https://github.com/backprop64/DAMM), [Nature Sci Rep](https://www.nature.com/articles/s41598-024-72367-2)
- DAMM was benchmarked against DeepLabCut SuperAnimal-TopViewMouse, another zero-shot rodent model (keypoints). — [DAMM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305), [SuperAnimal-TopViewMouse HF](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)

### Inferences
- Rough CPU tiers at 640x480 (inference only, not benchmarked by us):
  - Real-time-ish (tens of ms per frame): YOLOE-11/26 n/s, after you set text classes or visual prompts and fuse them.
  - About 1-10 s per frame: OWLv2-B at 960, Grounding DINO-T, Florence-2-base.
  - Tens of seconds per frame: SAM 2 automatic mode and SAM 3.
- What these tiers mean for a 10-min, 19 fps video (11,400 frames):
  - Only YOLOE or a distilled tiny YOLO can process every frame on CPU in reasonable time.
  - SAM 2 video propagation, even at about 1 s per frame for tiny with a single object, would take about 3 hours per video.
- Licence summary for commercial use:
  - OK: Florence-2 (MIT), OWLv2, original Grounding DINO and SAM 2 (Apache, per background knowledge).
  - Needs Enterprise licence or open-sourcing: Ultralytics YOLOE/YOLO-World (AGPL).
  - Not OK: DAMM (non-commercial).
  - Needs review: SAM 3 (custom licence).
  - Commercial terms unclear: T-Rex2 / GD1.5 (API only).

### Gaps
- I found no published CPU latency numbers at 640x480 for YOLOE visual-prompt mode, OWLv2 image-guided mode, Grounding DINO or Florence-2. These need a local benchmark.
- Several licences (SAM 2, SAM 3, OWLv2, Grounding DINO, Ultralytics AGPL) were not re-fetched this session.

## Q2. Visual-prompt / one-shot approaches (YOLOE VP, OWLv2 image-guided, T-Rex2) on small, low-contrast animals

### Takeaway
I found no published evaluation of YOLOE visual prompts, OWLv2 image-guided detection or T-Rex2 on lab rodents or comparable small, low-contrast animals. The papers only claim that visual prompts help with rare or hard-to-name categories, which fits this case. Whether one example box can find a 30 px white mouse on a white floor has to be tested empirically on our video.

### Cited Findings
- YOLOE paper: visual prompts help "less frequent objects that text prompts often struggle to accurately describe". Visual-prompt AP is close to text-prompt AP on LVIS (e.g. YOLOE-26x 38.5 vs 40.6). — [YOLOE arXiv](https://arxiv.org/pdf/2503.07465), [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- OWLv2 image-guided mode embeds a query image crop and matches it against image patches. The default NMS of 0.3 and the 960 px resize mean a 30 px object in a 640 px frame becomes about 45 px. — [HF OWLv2 docs](https://huggingface.co/docs/transformers/model_doc/owlv2)
- On a different domain (aerial imagery), open-vocabulary detectors degrade sharply compared with their natural-image benchmarks. — [arXiv 2601.22164](https://arxiv.org/html/2601.22164v1)
- Domain-trained DAMM generalises zero-shot to unseen mouse setups at 95.9-99.5% Mask AP75 but fails on unusual optics (26.9% on fisheye). This suggests mouse-specific training beats generic prompts, and that a few shots (20) fix most failures. — [DAMM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305)

### Inferences
- Low contrast (white on white, grayscale) removes the colour cue that CLIP-style text and visual embeddings rely on. Both text and visual prompts will likely struggle; the visual prompt has a better chance because it encodes the actual appearance.
- For YOLOE, take the visual prompt from the same video (`refer_image` = a frame showing the mouse) and lower `conf`. Expect false positives on wall corners and shadows.
- Temporal priors (one object, smooth motion, a static background) are a cheap way to boost any weak detector. Examples: background subtraction to propose ROIs, then classify crops.

### Gaps
- There is no quantitative data on visual-prompt detectors for objects of about 30 px at low contrast; this needs our own test.
- I found no rodent-specific T-Rex2 examples; T-Rex Label (an annotation tool) may be usable for labelling but was not evaluated.

## Q3. SAM 2 video propagation from one click: small objects, occlusion, long videos on CPU; lab-animal papers

### Takeaway
SAM 2 is the most robust "click once, track" option, and SAM-family tools are already used in animal-behaviour pipelines (SAMBA for flies and fish, DAMM's annotation GUI). Its known weaknesses match our case: drift under occlusion or disappearance, poorer tracking of tiny objects, and heavy compute. On CPU, a 5-10 min video is an hours-long offline job. I found no peer-reviewed SAM 2 mouse-tracking benchmark.

### Cited Findings
- SAMBA (bioRxiv 2025) uses SAM 2 with point prompts on one first frame to track Drosophila larvae/adults and zebrafish larvae; **no rodents were tested**. Runtime is ~20 min per 3-min video (5 larvae) on an A100 and ~30 min on an L4; a free T4 cannot finish within Colab's 2-hour limit. ID switches happen on collisions. Code: github.com/johnsonflygroup/SAMBA. — [SAMBA bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.21.671669v1.full)
- SAM 2 "can drift under fast or nonlinear motion, occlusions, background clutter, and target disappearance or reappearance" in long sequences. HiM2SAM, DAM4SAM/SAM2.1++ and others are training-free fixes. — [HiM2SAM summary](https://www.emergentmind.com/topics/him2sam), [DAM CVPR 2025 supp](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Videnovic_A_Distractor-Aware_Memory_CVPR_2025_supplemental.pdf)
- Datasets "characterized by extremely tiny to small objects and minimal inter-frame bounding box overlap significantly impacted tracker performance" for SAM2-based trackers (FMOX benchmark). — [FMOX review](https://www.themoonlight.io/review/benchmarking-sam2-based-trackers-on-fmox)
- Memory attention, not the image encoder, is SAM 2's main latency bottleneck. EdgeTAM runs at 16 FPS on an iPhone 15 Pro Max. — [EdgeTAM search snippet via EdgeDAM arXiv 2603.05463](https://arxiv.org/pdf/2603.05463)
- Ultralytics CPU benchmark: SAM2-t at about 23 s per image (M4 Air, probably segment-everything mode). — [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2/)
- A search snippet mentioned SAM 2 used "for semantic segmentation on mice in open field studies" with box prompts, and Track Anything (TAM) initialised with one click per mouse. I could not locate or verify the primary source (UNVERIFIED). One candidate bioRxiv PDF turned out to be a C. elegans paper. — [search result, unverified]
- DAMM uses SAM, not SAM 2, inside its annotation GUI (foreground/background clicks) to make mouse masks quickly. — [DAMM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305)

### Inferences
- With a 30 px mouse, SAM 2 at its native 1024 input upsamples the frame about 1.6x, which helps. Rearing against white walls (low contrast, shape change) is the likeliest failure; add re-prompts or negative clicks on the wall.
- Practical CPU use: run SAM 2.1-tiny on a subsampled set of frames, or on short clips, to produce training labels rather than to track the full video. Chunking the video (resetting memory every N frames with a box from the previous chunk) limits RAM growth.
- EdgeTAM or EfficientTAM variants may be worth a CPU test as faster SAM 2-style trackers (not verified on CPU here).

### Gaps
- I found no published accuracy numbers for SAM 2 or SAM 3 tracking single lab mice in open field or EPM.
- I found no published SAM 2.1-tiny CPU fps for single-object video propagation at 640x480.

## Q4. Practical pipeline: promptable model → auto-label → train a tiny YOLO

### Takeaway
"Distil a foundation model into a small YOLO" is an established, tooled workflow: Autodistill (Apache-2.0) and Ultralytics `auto_annotate`. DAMM's results show that about 20 labelled frames from a new setup fix most domain gaps. For our video, the pragmatic path is to bootstrap a few hundred labels with a promptable model or SAM 2 click-tracking, correct them by hand, and train YOLO11n/26n. That gives tens of ms per frame on CPU.

### Cited Findings
- Autodistill labels unlabelled images with a foundation base model, then trains a small target model. Base models include Grounded SAM / Grounded SAM 2, GroundingDINO, OWL-ViT/OWLv2 and DETIC; targets include YOLOv8/v5, YOLO-NAS and DETR. Licence is Apache-2.0. Example: `GroundedSAM(ontology=CaptionOntology({...})).label(...)`, then `YOLOv8("yolov8n.pt").train(...)`. — [autodistill GitHub](https://github.com/autodistill/autodistill)
- Ultralytics `auto_annotate()` combines a detector with SAM 2 to create segmentation labels. — [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2/)
- YOLOE docs: zero-shot accuracy is well below that of a model trained on your classes. — [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe/)
- DAMM few-shot: 56.4% → 84.7% Mask AP75 with 20 examples in hard settings. It suggests a small amount of in-domain labelling is very effective (Mask R-CNN). — [DAMM PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393305)
- YOLO11n-seg runs at about 24 ms per image on an M4 Air CPU. — [Ultralytics SAM 2 docs](https://docs.ultralytics.com/models/sam-2/)

### Inferences (suggested pipeline)
1. Seed the labels. Pick one:
   - (a) Click the mouse once in frame 0 and run SAM 2.1-tiny (Ultralytics `SAM2VideoPredictor`) on a 300-600 frame clip, or on several short clips spread across the video.
   - (b) Run YOLOE with a visual prompt (an example box from the same video) at low confidence, keeping the top-1 box per frame.
   - (c) Run DAMM zero-shot. This is for internal or research use only, because of its non-commercial licence.
2. Keep one box per frame and enforce temporal consistency (max displacement, size). Hand-fix the failures, especially rearing and wall frames, in CVAT, Label Studio or the DAMM SAM GUI.
3. Train YOLO11n or YOLO26n at imgsz 640 (or crop and upsample) on a few hundred frames, including hard negatives such as empty arena frames.
4. At runtime on CPU, run the tiny YOLO at tens of ms per frame, add a single-object Kalman/ByteTrack smoother, and fall back to background subtraction when confidence is low.
- On grayscale input, replicate the channel to 3 channels; foundation models expect RGB.

### Gaps
- I found no published example of this exact distillation pipeline (foundation model → tiny YOLO) for lab mice specifically. DAMM is the closest analogue (SAM-assisted labelling → Mask R-CNN).
- Whether YOLOE or OWLv2 produce usable pseudo-labels on our white-on-white footage is untested.
