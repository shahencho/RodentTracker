# Pretrained rodent (lab mouse / rat) detection & pose models on Roboflow Universe and Hugging Face

Research date: 2026-10-08. Method: web search, Hugging Face Hub REST API (`/api/models`, `/api/datasets`, raw READMEs and file listings, queried directly), Roboflow docs. **Caveat: every universe.roboflow.com page returned HTTP 403 to automated fetches (WebFetch and curl, including the search page), and no browser was available.** All Roboflow Universe project details below therefore come from search-engine snippets only. Project URLs appeared in search results, but their contents (sample images, mouse colour, camera view, mAP, licence) were **not opened or checked** unless stated otherwise.

## Q1. Which Roboflow Universe projects have a trained mouse/rat model, can the weights be downloaded, and what are the licences?

### Takeaway
Roboflow Universe has many small (about 80 to 1,800 image) community "lab mouse", "mice" and "rat" projects with a hosted "Roboflow 3.0 Object Detection (Fast)" model, plus a few keypoint projects. None are curated or benchmarked. In practice their weights cannot be downloaded as .pt from a Universe project. You can call them with an API key through Roboflow's serverless API, or through a self-hosted `inference` server on localhost, which pulls the weights using the key. That works on CPU but ties you to a Roboflow account. They are mainly useful as **datasets** to download (CC BY 4.0 is common) and fine-tune your own YOLO on.

### Cited Findings
**Candidate Universe projects (lab rodents, not computer mice)**
- **lab_mouse_detection** (workspace `mousescratch`): Roboflow 3.0 Object Detection (Fast), trained on 144 images (dataset version 2), published 12 Apr 2024 — [Roboflow Universe](https://universe.roboflow.com/mousescratch/lab_mouse_detection) (search snippet)
- **LaboratoryMice** (`uah-project`): Roboflow 3.0 OD (Fast), 165 images (version 4), published 4 Jul 2023 — [Roboflow Universe](https://universe.roboflow.com/uah-project/laboratorymice) (snippet)
- **lab-mouse** (`nervergiveup`): object detection with classes "borders" and "lab-mouse" — [Roboflow Universe](https://universe.roboflow.com/nervergiveup/lab-mouse-pqsvx) (snippet)
- **open field test** (`muza-2wj2q`): Roboflow 3.0 OD (Fast), 246 images (version 1), published 13 Jul 2024. This is the most directly relevant to open-field tracking by name — [Roboflow Universe](https://universe.roboflow.com/muza-2wj2q/open-field-test) (snippet)
- The search engine's summary called the models from the first search CC BY 4.0. That claim is per the search summary and was **not verified per project** — [search results incl. above URLs](https://universe.roboflow.com/muza-2wj2q/open-field-test)
- **Mouse Classifier** (`labgymtestsjm`): the name suggests LabGym-related use; it has a trained model with a results page — [Roboflow Universe](https://universe.roboflow.com/labgymtestsjm/mouse-classifier) (snippet; details unverified)
- **Mice Detection** (`hunter-diminick`) — [Roboflow Universe](https://universe.roboflow.com/hunter-diminick/mice-detection-flgeh) (trained model per snippet; details unverified)
- **Mouse detection** (`first-yug0v`) — [Roboflow Universe](https://universe.roboflow.com/first-yug0v/mouse-detection-p2nn4); **Mouse detection** (`nan56`) — [link](https://universe.roboflow.com/nan56/mouse-detection-ehwvy) (unverified content)
- **mice** dataset (`meg-lab`) — [Roboflow Universe](https://universe.roboflow.com/meg-lab/mice-teqzt) (dataset; unverified content)
- **mice model** (`titan-jk1qh`): 974 images plus a pre-trained model and API — [Roboflow Universe](https://universe.roboflow.com/titan-jk1qh/mice-model) (snippet)
- **Mouse Finder** (`yolo-test-pstn6`): 1,763 images plus a pre-trained model. This is the largest found, but it is unclear whether it shows lab mice or wild or house mice — [Roboflow Universe](https://universe.roboflow.com/yolo-test-pstn6/mouse-finder) (snippet)
- **Mice** (`mice-durxn`): 113 images, 3 classes (body, mice, tail) — [Roboflow Universe](https://universe.roboflow.com/mice-durxn/mice-rq13l) (snippet)
- **Mouse** (`animal-aikuw`): 83 images plus a pre-trained model — [Roboflow Universe](https://universe.roboflow.com/animal-aikuw/mouse-wggrw) (snippet)
- Other "mouse" projects that showed up in search, with unknown subject (the animal may not be a lab mouse): [yolo-nffya/mouse-gvgo9](https://universe.roboflow.com/yolo-nffya/mouse-gvgo9), [test-fxbet/mouse-wbji6](https://universe.roboflow.com/test-fxbet/mouse-wbji6), [mouse-r75v1/mouse-qabew](https://universe.roboflow.com/mouse-r75v1/mouse-qabew), [naranbat/mouse-hduvv](https://universe.roboflow.com/naranbat/mouse-hduvv), [qhuy/mouse-dgjtb](https://universe.roboflow.com/qhuy/mouse-dgjtb)
- Universe search page filtered to "class:mice" with trained models, useful for manual browsing — [Roboflow Universe search](https://universe.roboflow.com/search?q=class%3Amice+object+detection+trained+model)

**Keypoint / pose projects**
- **RodentBehavioral** (`rodentdetection-d341g`): Roboflow 3.0 **Keypoint Detection**, trained on 144 images from a COCO checkpoint, updated 10 Dec 2024 — [Roboflow Universe](https://universe.roboflow.com/rodentdetection-d341g/rodentbehavioral) (snippet)
- **mouse keypoint** dataset (`project-ca7wy`) — [Roboflow Universe](https://universe.roboflow.com/project-ca7wy/mouse-keypoint-ba3nf); **Pose** (`social-interaction-nurrf`), a mouse pose dataset — [link](https://universe.roboflow.com/social-interaction-nurrf/pose-tjand); a "Micepose" keypoint project with 600 images was mentioned in the search summary, but **no URL was returned (unverified)**
- **tt_2026** (`ratones`, Spanish for "mice") dataset — [Roboflow Universe](https://universe.roboflow.com/ratones/tt_2026) (content unverified)

**Rat projects**
- **rat_detection_v1** (RaVision), dataset, CC BY 4.0 — [Roboflow Universe](https://universe.roboflow.com/ravision/rat_detection_v1) (snippet)
- **rat_detection** (`frauas`), model — [Roboflow Universe](https://universe.roboflow.com/frauas/rat_detection)
- **Rat Detection** (`saeed-zekkt`), model with a "dark-mice" class — [Roboflow Universe](https://universe.roboflow.com/saeed-zekkt/rat-detection-gdpbn)
- Other rat datasets: [ashveer-test-workspace/rat-detection-j9gzi](https://universe.roboflow.com/ashveer-test-workspace/rat-detection-j9gzi), [mbdcomputervision/rat-detection2](https://universe.roboflow.com/mbdcomputervision/rat-detection2). These are likely pest or wild rats and are unverified.
- **Elevated plus maze:** no Roboflow Universe project specific to the EPM turned up in two searches ([search 1](https://avesis.ktu.edu.tr/yayin/64c31c53-e2e9-4a3f-b5e6-a87ac7dfe1c9/elevated-plus-maze-on-based-computer-vision) returned only a non-Roboflow paper on rat EPM tracking with a Hough transform plus Otsu thresholding).

**Weight download and local inference**
- Roboflow docs: "Core, including the Free Tier, supports downloading weights through the Python SDK" for models trained in **your own workspace**. Supported architectures include Roboflow 3.0, YOLOv8, YOLOv12, YOLO26, YOLO-NAS and RF-DETR. SAM3 and RF-DETR NAS weights cannot be downloaded — [Roboflow docs: Download model weights](https://docs.roboflow.com/deploy/download-roboflow-model-weights)
- The docs page does not say that Universe models (other people's projects) can be downloaded. A June 2026 community-forum answer (from a community member, not staff) states: "No — Roboflow does not support downloading model weight files (.pt or otherwise) directly from Roboflow Universe projects" — [Roboflow forum thread](https://discuss.roboflow.com/t/how-can-i-get-the-native-pt-file-of-the-https-universe-roboflow-com-cricket-qnb5l-basketball-xil7x-model/12294/3)
- Universe models can be run in code through the serverless hosted API (`https://serverless.roboflow.com`) or a self-hosted inference server at `http://localhost:9001`. Both need `ROBOFLOW_API_KEY` — [Roboflow docs: Run a Universe model in code](https://docs.roboflow.com/datasets/universe/universe/find-a-model-on-universe)
- Universe model pages also offer in-browser test inference (upload an image or video, or use a webcam) and generate code snippets — [Roboflow blog: YOLOv8 models on Universe](https://blog.roboflow.com/yolov8-models-apis-datasets/)

### Inferences
- The fastest no-training test is to upload 2 or 3 frames of our video to the "Try this model" widget on each candidate's model page (open-field-test, lab_mouse_detection, LaboratoryMice, mice model, Mouse Finder, RodentBehavioral). This costs nothing and needs no code.
- Roboflow 3.0 "Fast" models trained on 144 to 250 images rarely generalise to a different arena, mouse colour or camera. Expect failures similar to stock COCO unless a project happens to contain white mice on a light background.
- The best practical use of Universe is probably to download the datasets (CC BY 4.0 where stated), merge them with about 100 to 200 labelled frames of our own video, and train YOLO11n locally.

### Gaps
- For every Universe project: sample-image content (mouse colour, arena colour, top-down vs side view, IR or greyscale), reported mAP and exact licence could **not be verified**, because Roboflow returned 403 to automated access and no browser was connected. A human should open each link.
- Whether the local `inference` server caches Universe weights for later offline CPU use was not confirmed in the docs I read.

## Q2. Which Hugging Face repos host YOLO or other lightweight mouse/rat detectors or pose models, with downloadable weights, and under what licence?

### Takeaway
Hugging Face has very few real lab-rodent vision models, because most "mouse" hits are genomics models, Mickey Mouse LoRAs or computer-mouse models. Four were verified by file listing:
- **DeepLabCut SuperAnimal-TopViewMouse:** top-down pose, includes small detectors, trained on 5K images that include white mice and EPM/OFT data. This is the strongest candidate.
- **KUNIN/mice-yolo-inf:** YOLOv8n-pose for mice, with OpenVINO int8 builds that suit CPU.
- **KUNIN/mice-gait:** bottom-view gait models, likely irrelevant.
- **kaopanboonyuen/DeepRodent:** an Ultralytics .pt with segmentation and oriented boxes, from a single author with a private dataset. Treat with caution.

### Cited Findings
**1. mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse (verified)**
- Files: `superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt` (76 MB), `..._fasterrcnn_resnet50_fpn_v2.pt` (173 MB), `..._rtmpose_s.pt` (23 MB), `..._rtmpose_m.pt` (56 MB), `..._rtmpose_x.pt`, `..._hrnet_w32.pt` (118 MB), `..._hrnet_w48.pt`, `..._resnet_50.pt`, `..._resnet_101.pt`, `detector.pt` (518 MB), `pose_model.pth`, and a DLCRNet TF tarball — [HF API file listing](https://huggingface.co/api/models/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Model card: intended "for pose tracking of lab mice videos filmed from an overhead view"; usable "plug-and-play" when extremely high precision is not required. It is not suitable for other camera views — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Training data (TopViewMouse-5K) includes **3CSI, BM, EPM, LDB, OFT** (Sturman et al.), BlackMice, **WhiteMice (from SimBA / Golden lab)**, TriMouse, DLC-Openfield, Kiehn-lab open field, and MausHaus (C57BL/6J at 640x480) — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Caveats in the card: "this dataset only has one set of white mice, therefore it may not generalize well to diverse settings of white lab mice"; the data is mainly C57BL/6J mice with some CD1. Lighting and contrast are named as robustness factors — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Keypoints: a 27-point unified top-view vocabulary (details in Ye et al.) — [arXiv 2203.07436](https://arxiv.org/abs/2203.07436). The exact count is from my memory of the paper, **not checked in this session**; the card says MausHaus has 26 annotated keypoints.
- Licence: **Modified MIT, academic and non-commercial use only**; commercial use needs a licence from the Mathis lab or the EPFL technology transfer office — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Loader: `dlclibrary.download_huggingface_model("superanimal_topviewmouse_model", ...)` — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Community ONNX conversion: `JCobosAlvarez/DeepLabCut-TopViewMouse-onnx` holds `hrnet_w32.zip` (106 MB) and has no README or licence — [HF API](https://huggingface.co/api/models/JCobosAlvarez/DeepLabCut-TopViewMouse-onnx). It is unverified, and the upstream non-commercial licence presumably still applies.

**2. KUNIN/mice-yolo-inf (verified files; sparse documentation)**
- Files: `models/kunin-mice-pose.v0.1.4.pt` and `v0.1.5n.pt` (about 5.8 MB each, consistent with YOLOv8n-pose), plus **OpenVINO fp32, fp16 and int8 exports** (int8 .bin is about 3 MB), a FastAPI `app.py`, and a 376 MB Windows `app.exe` — [HF API file listing](https://huggingface.co/api/models/KUNIN/mice-yolo-inf)
- Licence tag: **CC BY 4.0**. The README is empty apart from that tag — [HF repo](https://huggingface.co/KUNIN/mice-yolo-inf)
- `app.py` loads `YOLO("models/kunin-mice-pose.v0.1.5n_int8_openvino_model", task='pose')`, so it is an Ultralytics pose model meant for CPU or OpenVINO — [app.py](https://huggingface.co/KUNIN/mice-yolo-inf/raw/main/app.py)
- The sibling repo KUNIN/single-mouse-webrtc-pose has the same `kunin-mice-pose.v0.1.5n.pt` and documents it (in Chinese) as a **YOLOv8 single-mouse pose model with 8 keypoints**: nose, left ear, right ear, body, left leg, right leg, tail base, tail tip. No licence is declared — [README](https://huggingface.co/KUNIN/single-mouse-webrtc-pose)
- That repo's demo image is a **synthetic cartoon mouse on a grey background**, which says nothing about the training data. Training data, camera view, mouse colour and mAP are all undocumented — [demo image](https://huggingface.co/KUNIN/single-mouse-webrtc-pose/resolve/main/demo_visualization_result.jpg)

**3. KUNIN/mice-gait (verified files)**
- Files: `mice-gait-v1.0.{pt,onnx,engine,mlpackage}` and `mice-pose-bottomview-v1.0.{pt,onnx,...}` (.pt about 5.5 MB; ONNX about 11 MB). The model is **bottom-view** (catwalk/gait), there is no licence and no README — [HF API](https://huggingface.co/api/models/KUNIN/mice-gait)

**4. kaopanboonyuen/DeepRodent (verified files)**
- Files: `DeepRodent_WEIGHT.pt` (6.8 MB, YOLO-n-sized), Apache-2.0, `library_name: ultralytics`, created 1 Jul 2026 — [HF API](https://huggingface.co/api/models/kaopanboonyuen/DeepRodent)
- Claims: detection plus OBB plus instance segmentation plus temporal embedding. mAP50 is 96.2 and mAP50-95 is 84.6 on a **private** dataset of about 30K frames ("multi-setting laboratory"), at 154 FPS (GPU assumed). Usage is `YOLO("DeepRodent_WEIGHT.pt").predict(...)` — [Model card](https://huggingface.co/kaopanboonyuen/DeepRodent)
- Sample images in the companion Space show a dim, purple-tinted IR-like cage interior from an oblique, side-ish view, not a top-down open field — [Space sample image](https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo/resolve/main/rodent_sample_images/rodent_sample_images_01.jpg). The Space also contains an `assets/architecture_placeholder.svg` — [Space file list](https://huggingface.co/api/spaces/kaopanboonyuen/DeepRodent-Demo)

**Repos checked and ruled out**
- `nicsap/miceDetector`: empty, holding only `.gitattributes` — [HF API](https://huggingface.co/api/models/nicsap/miceDetector)
- `Biomed-imaging-lab/AI-mouse-detector`: empty, README only, Apache-2.0 — [HF API](https://huggingface.co/api/models/Biomed-imaging-lab/AI-mouse-detector)
- `mulemp/Lab_Mouse`: gated, and contains anime-style LoRA safetensors, so it is not a rodent model — [HF API](https://huggingface.co/api/models/mulemp/Lab_Mouse)
- `Nomi78600/keyboard-mouse-detection-yolov8` and `tokakimiku/computer-mouse` are **computer-mouse** models and irrelevant — [HF search](https://huggingface.co/api/models?search=mouse-detection)
- `paninski-lab/vitb16-beast-mirror-mouse`: a 1.36 GB ViT-B BEAST checkpoint for a mirror-mouse (head-fixed) dataset, CC BY 4.0. It is not a detector and too heavy for CPU — [HF repo](https://huggingface.co/paninski-lab/vitb16-beast-mirror-mouse)
- `jpoberhauser/molmo2-4b-mice-lora`: a 4B-parameter VLM LoRA with an empty model card, so far too heavy for CPU tracking — [HF repo](https://huggingface.co/jpoberhauser/molmo2-4b-mice-lora)
- `EtherealHorizon/depth_anything_rodent`: a ZoeDepth checkpoint (1.3 GB), MIT, depth estimation only — [HF API](https://huggingface.co/api/models/EtherealHorizon/depth_anything_rodent)

**Hugging Face datasets worth knowing**
- `gauravkaul/DAMM_mouse_detection`: the datasets behind the **Detect Any Mouse Model (DAMM)**. They are COCO-style boxes plus polygon masks: an "AER pretraining" set of 2,209 images plus about 100-image sets covering playground, light_dark_nest, perfect_setup, colorful_multi_mice, black_multi_mice, golden_open_field, mars_multi, a McDannald rat operant chamber, and more. Detection and tracking zips are included; no licence is declared on HF — [HF dataset card](https://huggingface.co/datasets/gauravkaul/DAMM_mouse_detection)
- `zperlman/mouse-seg` and `mouse-seg2`: small semantic-segmentation sets from a Lucid Vision camera, undocumented — [HF API](https://huggingface.co/api/datasets/zperlman/mouse-seg2)

### Inferences
- **SuperAnimal-TopViewMouse is the most credible zero-shot option.** It is the only one trained on many labs' top-view data, including EPM, OFT and a white-mouse set, and it is maintained by the DeepLabCut team. On CPU, pair the Faster R-CNN MobileNetV3 detector (76 MB) with RTMPose-S (23 MB) through DeepLabCut 3.x's PyTorch engine. That should be workable at reduced frame rate on 640x480 frames, but **CPU fps was not measured**.
- KUNIN's YOLOv8n-pose int8 OpenVINO model is the cheapest to try on CPU (about 3 MB, directly loadable by Ultralytics). Its training domain is unknown, so it is a lottery ticket.
- DeepRodent is easy to try with `pip install ultralytics`. However, it comes from a single author, its claims rest on a private dataset, its samples come from a different domain (oblique IR cage), and its Space has placeholder assets. Treat the claims with scepticism until it is tested.
- DAMM itself is a Mask R-CNN (Detectron2) "detect any mouse" model built for zero-shot use across setups. Its weights appear to be hosted on GitHub or Google Drive rather than HF (**unverified**; not fetched this session). It may be worth a separate check, since cross-setup generalisation is its design goal.

### Gaps
- No HF repo documents a **white mouse on a white background** case explicitly.
- I found no YOLO11 lab-rodent model on HF.
- I found no rat-specific lab model on HF; the "rat" searches returned only noise.
- KUNIN models: training data, view and mAP are unknown.
- DeepRodent: the paper is not independently verified, and the classes and output heads inside the .pt were not inspected.
- DAMM weight hosting and licence were not verified this session.

## Q3. Which options are most likely to work on a small (~30 px) white mouse in a white arena, filmed from a tilted top-down view, on CPU?

### Takeaway
Nothing found is purpose-built for low-contrast white-on-white footage. Ranked by expected chance of zero-shot success:
1. SuperAnimal-TopViewMouse (detector plus RTMPose-S), because it was trained on multi-lab top-view OFT/EPM data and has some white-mouse data.
2. DAMM, which is built for generalisation (weights not verified).
3. The Roboflow "open field test" and other lab-mouse models, tested in the browser widget.
4. KUNIN YOLOv8n-pose and DeepRodent, as cheap long shots.

If all of them fail, fine-tuning YOLO11n on a few hundred labelled frames will probably be needed, seeded with the DAMM and Universe datasets.

### Cited Findings
- SuperAnimal-TopViewMouse training mix includes EPM, OFT and a WhiteMice (SimBA) set. Its card warns that white-mouse generalisation may be weak and that lighting and contrast affect robustness — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- The card recommends "video adaptation" (self-supervised adaptation on your own video) or fine-tuning if zero-shot performance is insufficient — [Model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- DeepRodent sample imagery comes from an oblique IR cage view, which is a different domain from a bright white open field — [Space sample](https://huggingface.co/spaces/kaopanboonyuen/DeepRodent-Demo/resolve/main/rodent_sample_images/rodent_sample_images_01.jpg)
- The DAMM datasets span many cage and arena setups, including open-field and multi-colour mice — [HF dataset card](https://huggingface.co/datasets/gauravkaul/DAMM_mouse_detection)

### Inferences
- A ~30 px target at 640x480 is small for detectors trained at a 640 input. Upscaling or tiling a crop of the arena, for example running at an imgsz of 960 to 1280 on the arena ROI, may help any of these models more than switching models does.
- White-on-white is a contrast problem. Contrast stretching or CLAHE before inference, plus background subtraction (the classic approach used in non-Roboflow EPM work), could rescue weak detections. A hybrid design is plausible: background subtraction for the position track, and a model only for disambiguation or keypoints.
- If any pretrained model gets even partial hits, its outputs can bootstrap pseudo-labels for a quick YOLO11n fine-tune, which runs fast on CPU at inference time.

### Gaps
- No source reports benchmark results for white mice on white backgrounds for any of these models.
- No CPU throughput numbers were found for SuperAnimal RTMPose-S plus the MobileNet detector, for the KUNIN int8 model, or for DeepRodent.
- None of the models was actually run on the user's video in this session.
