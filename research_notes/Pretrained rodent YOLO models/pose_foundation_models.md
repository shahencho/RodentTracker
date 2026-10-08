# Pretrained / zero-shot animal pose & detection model families for lab mice (as of Oct 2026)

Context: single white mouse (~30 px) in a white arena, 640x480 grayscale, 19 fps, tilted top-down camera, frequent wall-hugging/rearing. Goal: per-frame body centre (+ nose / tail base) with no labelling or training, on CPU, ideally commercially licensable.

## Q1. SuperAnimal-TopViewMouse (DeepLabCut): data, keypoints, detector, backbones, accuracy, video adaptation, download, licence

### Takeaway
SuperAnimal-TopViewMouse is the only mature, downloadable, zero-shot top-view lab-mouse pose model. It ships with Faster R-CNN detectors (ResNet50-FPN-v2 and a lighter MobileNetV3-Large-FPN) and several pose heads (HRNet-w32/w48, ResNet-50/101, RTMPose-s/m/x). It includes some white-mouse data, but the authors themselves list white mice and atypical lighting as weak spots. The licence is **research / non-commercial only**, and a commercial licence has to be negotiated with EPFL-TTO.

### Cited Findings
**Training data**
- HF model card: trained on a combination of datasets, namely 3CSI, BM, EPM, LDB, OFT (behavioural-test arenas, including **elevated plus maze and open field**), BlackMice, **WhiteMice (courtesy of Prof. Sam Golden)**, TriMouse, DLC-Openfield, Kiehn-Lab (openfield, swimming, treadmill) and MausHaus. MausHaus was recorded at 30 Hz, 640x480, with White Matter eV cameras. The card says the mice are "primarily C57BL/6J … limited CD1 examples; one set of white mice dataset". — [HF model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- The paper (arXiv v4) describes SA-TVM as roughly 5,000 images merged from multiple overhead lab-mouse datasets into a unified keypoint vocabulary. The fetched summary said "13 datasets", while the HF card lists about 9 named groups, so the count depends on how sub-datasets are counted. Treat it as "~5K images, ~9–13 source datasets". — [arXiv 2203.07436v4](https://arxiv.org/html/2203.07436v4); [Zenodo dataset record](https://zenodo.org/records/10618947)
- Limitations stated by the authors and the model card: reduced accuracy under atypical lighting, **poor generalisation to diverse white-mouse populations**, annotator bias, and a mostly C57BL/6J population. The paper also names IR imaging as a domain shift. — [HF model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse); [arXiv](https://arxiv.org/html/2203.07436v4)

**Keypoints**
- The sources disagree on the count. The DLC Model Zoo docs and the paper say **27 keypoints**. The HF card and a DLC README snippet say 26. — [DLC Model Zoo docs](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html); [arXiv](https://arxiv.org/html/2203.07436v4); contradicted by [HF card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- (UNVERIFIED, from memory) The keypoint names include nose, ears and ear tips, eyes, neck, head_midpoint, mid_back, **mouse_center**, mid_backend(1–3), **tail_base**, tail1–tail5/tail_end, and left/right shoulder, midside and hip. If this is right, a centre point, nose and tail base can be read directly from the output. Check against the model's config yaml after download.

**Detector and pose backbones (files on HF, verified listing)**
- `superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt` (173 MB) and `superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt` (76 MB) are the detectors. Legacy `detector.pt` (518 MB) is also Faster R-CNN. — [HF file tree](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse/tree/main)
- Pose heads: `hrnet_w32` (118 MB), `hrnet_w48` (262 MB), `resnet_50` (100 MB), `resnet_101` (171 MB), `rtmpose_s` (23 MB), `rtmpose_m` (56 MB), `rtmpose_x` (201 MB), plus legacy TF bottom-up DLCRNet (`DLC_ma_supertopview5k_resnet_50…tar.gz`). — [HF file tree](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse/tree/main)
- The DLC docs describe the top-down HRNet model as paired with "a trained ResNet50-based Faster-RCNN". They also show the API call: `deeplabcut.video_inference_superanimal([video], "superanimal_topviewmouse", model_name="hrnet_w32", detector_name="fasterrcnn_resnet50_fpn_v2", video_adapt=False)`. — [DLC Model Zoo docs](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)
- The **lightest CPU-friendly combination** is `rtmpose_s` with `fasterrcnn_mobilenet_v3_large_fpn`. A model-zoo table snippet reports rtmpose_s at **92.9 mAP on SA-TVM DLC-OpenField**. I saw this only in a search snippet of a mirrored DLC doc, so it is unverified on the primary page. — [DLC Model Zoo (mirror snippet)](https://deeplabcuthtbprolgithubhtbprolio-s.evpn.library.nenu.edu.cn/DeepLabCut/docs/ModelZoo.html)

**Zero-shot accuracy**
- On the held-out DLC-Openfield set, zero-shot RMSE was about **4.9 px for HRNet-w32**, 4.57 px for AnimalTokenPose and 14.3 px for DLCRNet. The paper notes a mouse nose is about 10 px wide in that data. — [arXiv 2203.07436v4](https://arxiv.org/html/2203.07436v4)
- The DLC docs give a `scale_list` parameter for mismatches in animal size, e.g. `range(200, 600, 50)` for animals of about 400 px. This matters for our ~30 px mouse. — [DLC Model Zoo docs](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)

**Video adaptation**
- `video_adapt=True` runs unsupervised self-training on the user's own video with pseudo-labels, which "minimally extend[s]" processing time and reduces jitter (F(1,23286)=190.03, p<.0001). The paper reports about 12 FPS during adaptation vs about 40 FPS baseline inference, which I assume is on GPU. — [DLC docs](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html); [arXiv](https://arxiv.org/html/2203.07436v4)

**Licence**
- HF card: "Modified MIT … academic, non-commercial purposes only", plus a no-harm-to-animals clause. — [HF model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse); [Zenodo](https://zenodo.org/records/10618947)
- DLC GitHub README: the code is **LGPL-3.0**, but "SuperAnimal models are provided for research use only (non-commercial use)." — [DeepLabCut GitHub](https://github.com/DeepLabCut/DeepLabCut)
- Contact EPFL-TTO for a full commercial licence. — [DeepLabCut GitHub (search snippet)](https://github.com/DeepLabCut/deeplabcut)
- Conflict: the fetched arXiv summary said "Models: CC BY-SA 4.0". This likely refers to the paper text or an old version. The current model cards and README are explicit that use is non-commercial, so treat that as binding. Note that SuperAnimal-Quadruped's card at one revision showed apache-2.0, which shows the cards have changed over time. — [arXiv](https://arxiv.org/html/2203.07436v4); [Quadruped card revision](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/blob/d69a44904d67dabc806af97f490acd8006129b8b/README.md)

**Install**
- DLC 3.x uses a PyTorch backend. The recommended install is PyTorch first, then `pip install deeplabcut`. I found no Windows-CPU-specific notes. — [DeepLabCut GitHub](https://github.com/DeepLabCut/DeepLabCut)

### Inferences
- It can technically be run as-is on CPU: install CPU PyTorch, `pip install deeplabcut`, and the weights auto-download from HF. On CPU at 640x480 I would expect roughly 2–10 fps with the Faster R-CNN ResNet50 detector plus HRNet-w32, and several times faster with MobileNetV3 plus RTMPose-s. These are my estimates and were not measured.
- Our video is hard for this model. A ~30 px mouse is far smaller than the training animals, so try upscaling or `scale_list`. On white-on-white, the detector is the likely failure point: the authors flag white mice and lighting as weaknesses. A workaround is to skip the DLC detector and feed pose crops from our own background-subtraction box. Another is to use only the detector box centre.
- For a commercial product, using SuperAnimal weights requires an EPFL-TTO licence. It is fine for internal R&D, for evaluating, or as a pseudo-labeller for research. Whether pseudo-labels made with it can be used to train a commercial model is a legal grey area and needs EPFL-TTO confirmation.

### Gaps
- No published CPU benchmarks were found. No accuracy figures for albino-on-white or for animals under ~50 px were found.
- The exact current keypoint list (26 vs 27) was not verified from the config yaml.
- The primary DLC model-zoo performance table for each detector/pose combination could not be fetched; mlabofai.org/dlc-modelzoo returned 404.

## Q2. SLEAP pretrained models

### Takeaway
SLEAP does not ship a general-purpose pretrained mouse model for zero-shot use. "Pretrained" in SLEAP means ImageNet backbones that you then train on your own labels. The public mouse datasets are bottom-view C57 open field and top-view Swiss Webster home cage. Their trained models (on OSF) are dataset-specific and unlikely to transfer to our setup.

### Cited Findings
- SLEAP advertises "pretrained and customizable neural network architectures that deliver accurate predictions with very few labels". It reports <200 labels for 90% of peak accuracy on mice. — [SLEAP paper (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9007740/); [PyPI](https://pypi.org/project/sleap/)
- The benchmark datasets are **mice_of** and **mice_hc**:
  - mice_of: C57BL/6J, imaged **from below**, open field, 1–5 mice, 2,950 frames, 11 nodes.
  - mice_hc: **white Swiss Webster**, top view, home cage, 2 mice, 2,948 frames, 5 nodes.
  - The datasets page lists no trained models and points to OSF (osf.io/36har) for the "full dataset and model collection".
  — [SLEAP legacy datasets page](https://legacy.sleap.ai/datasets.html)
- SLEAP code is BSD-3-Clause. (UNVERIFIED in this session, from prior knowledge.)

### Inferences
- The mice_hc model (white mice, top view, 5 nodes) is the closest SLEAP artefact to our case. It was trained on one home-cage rig with dark-ish bedding contrast, so zero-shot transfer to a tilted white arena is unlikely to be reliable. Its licence on OSF is unknown.

### Gaps
- I did not open the OSF repo to confirm which trained models exist or their licence.

## Q3. MMPose / AP-10K / AnimalPose / Animal Kingdom models on top-down lab mice

### Takeaway
These are trained on in-the-wild, mostly side-view photographs of larger animals. AP-10K does not clearly include mice. They are not suited to a ~30 px top-down lab mouse and are not worth testing as a primary path. Licences are permissive (MMPose code Apache-2.0, AP-10K data CC-BY-4.0).

### Cited Findings
- AP-10K has 10,015 images from 23 families and 54 species, with 17 keypoints (eyes, nose, neck, tail root, limbs). Its README does not mention rodents or mice. The data is **CC-BY-4.0**. Pretrained HRNet-w32/w48 and ResNet-50/101 weights are available via MMPose. — [AP-10K GitHub](https://github.com/AlexTheBad/AP-10K)
- HRNet-w48 reaches 0.728 AP on the AP-10K validation set at 256x256. — [AP-10K paper](https://arxiv.org/pdf/2108.12617)
- The SuperAnimal paper presents TopViewMouse precisely because quadruped/in-the-wild models do not cover the top-down lab view. — [arXiv 2203.07436](https://arxiv.org/html/2203.07436v4)

### Inferences
- From directly above, a mouse is a featureless ellipse with no visible legs. AP-10K-style keypoints (limbs, side-view) would be meaningless. Expected zero-shot success on our video is very low.
- MMPose's code (Apache-2.0, unverified this session) is commercially usable. It is useful only as a framework for training our own RTMPose/RTMDet model, not as a zero-shot model.

### Gaps
- The Animal Kingdom and AnimalPose licences were not checked. MMPose licence and RTMDet weight licences were not re-verified this session.

## Q4. Other pretrained top-view mouse models (MARS, Kumar lab, Lightning Pose)

### Takeaway
- **MARS** has downloadable detector and pose weights. They are tied to a specific rig (home cage, black resident plus white intruder pair, 9 top-view keypoints, old TensorFlow), so they are unsuitable here.
- **Kumar lab deep-hrnet-mouse** (HRNet-w32, 12 keypoints, JAX open-field rig) is the most relevant single-mouse open-field model. Its weights ship in a Singularity image and its training data is non-commercial.
- **Lightning Pose** has no zero-shot mouse model zoo.

### Cited Findings
**MARS**
- Detector: MSC-MultiBox with Inception-ResNet-v2. Pose: 8-stack Stacked Hourglass. 9 top-view keypoints (tail mid/end dropped). Footage was 1024x570 top view at 30 fps; IR is not required. Runs at about 11 Hz for two-animal detection plus pose on suitable hardware. — [MARS bioRxiv](https://www.biorxiv.org/content/10.1101/2020.07.26.222299v2.full)
- MARS expects "top-view videos featuring pairs of interacting mice, with a black resident mouse and a white intruder mouse", "standard home cage, ~30Hz, colour or grayscale". A GPU is recommended but not required. — [MARS GitHub](https://github.com/neuroethology/MARS)
- Trained detection/pose models and classifiers are distributed via CaltechDATA. — [CaltechDATA record](https://data.caltech.edu/records/1655)
- The eLife article itself is CC-BY. — [eLife](https://elifesciences.org/articles/63720)

**Kumar lab (JAX)**
- deep-hrnet-mouse is HRNet-W32 on 480x480 top-down open-field frames. It outputs 12 keypoints: nose, ears, base neck, 4 paws, **center spine, base tail**, mid tail and tip tail. — [Zenodo release](https://zenodo.org/records/6380163); [GitHub](https://github.com/KumarLabJax/deep-hrnet-mouse)
- The code is under the **MIT** licence, but the training data is released "under a Non-Commercial license". The weights/runtime are packaged in a Singularity image (`deep-hres-net-2019-06-28.simg`). — [GitHub](https://github.com/KumarLabJax/deep-hrnet-mouse); [Zenodo](https://zenodo.org/records/6380163)

**Lightning Pose**
- I found no pretrained or zero-shot mouse model zoo. It is a semi-supervised training framework that needs labels. — [Lightning Pose docs](https://lightning-pose.readthedocs.io/en/v2.3.0/_sources/source/user_guide_singleview/training.rst.txt)

### Inferences
- MARS: the pair-specific, black/white-identity design and the TF1-era stack make it a poor fit for a single white mouse. Expect detector failure on white-on-white.
- Kumar: the JAX rig has a fixed overhead camera, a uniform floor and 480x480 crops, and mice larger than 30 px. Transfer to a tilted 640x480 EPM/OF is uncertain. Singularity is Linux-only, so a Windows CPU run would mean extracting the checkpoint manually. The licence status of the weights is ambiguous: the code is MIT, but the weights were derived from non-commercial data.
- DANNCE is 3D multi-camera and out of scope.

### Gaps
- The MARS LICENSE file could not be fetched (404 at the raw path). The MARS weights licence and whether it has a non-commercial clause are unverified.
- The Kumar weights' explicit licence and their coat-colour coverage (the JAX strain survey included albino strains, unverified) were not confirmed.

## Q5. Commercial licence implications and likelihood of success on our hard video

### Takeaway
None of the mouse-specific zero-shot models are clearly licensed for commercial redistribution. SuperAnimal is explicitly non-commercial, with an EPFL-TTO commercial licence available. Kumar's training data is non-commercial, and the MARS and SLEAP OSF models are unverified. The permissively licensed models (AP-10K/MMPose) don't fit the task. For a sellable product, the realistic path is classical background subtraction or a self-trained small detector/pose model, with SuperAnimal used at most as an internal evaluation tool or licensed from EPFL.

### Cited Findings
- SuperAnimal: non-commercial; commercial licence via EPFL-TTO. — [DLC GitHub](https://github.com/DeepLabCut/DeepLabCut); [HF card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
- Kumar: code MIT, training data non-commercial. — [GitHub](https://github.com/KumarLabJax/deep-hrnet-mouse)
- AP-10K: CC-BY-4.0. — [AP-10K GitHub](https://github.com/AlexTheBad/AP-10K)
- SuperAnimal's authors flag white-mouse generalisation and lighting as weaknesses. — [HF card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)

### Inferences (likelihood on our video: 30 px white mouse, white arena, tilted camera, rearing)
- **SuperAnimal-TopViewMouse:** moderate at best, and the most likely of all candidates to work partially. It has EPM/OFT and white-mouse data in training. Failure risks are small size (upsample frames about 2–3x or use `scale_list`), low contrast for the Faster R-CNN detector, and rearing poses. Worth a 1-hour test on CPU with rtmpose_s plus mobilenet and with hrnet_w32 plus resnet50, both with `video_adapt=True`. If the detector fails, give the pose model crops from our own motion or background-subtraction box.
- **Kumar HRNet:** low to moderate. It is designed for a fixed JAX arena, and running it on Windows is awkward.
- **MARS:** low.
- **SLEAP mice_hc:** low.
- **AP-10K / AnimalPose:** very low.
- For the body centre alone, a tuned background-subtraction or frame-difference tracker probably beats all of these on licence, speed and robustness. Learned models mainly add nose and tail-base orientation.

### Gaps
- None of the models have been benchmarked on white-on-white or under-50-px mice. The actual result has to be measured on our test video.
