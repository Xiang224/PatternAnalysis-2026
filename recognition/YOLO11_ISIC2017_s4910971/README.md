# YOLO11 Skin Lesion Detection on ISIC 2017

COMP3710 / COMP7710 project by s4910971. This project will fine-tune a
COCO-pretrained YOLO11n detector to localize and classify dermoscopic lesions.
The three target diagnoses are melanoma, nevus, and seborrheic keratosis.

## Current status

Initial repository and reproducible asset preparation. Training has not started;
no model performance or clinical utility is claimed. The final training pipeline,
experiments, feasibility review and error analysis remain to be implemented.

## Data and model

- Official ISIC 2017 images: 2,000 training, 150 validation, 600 test.
- Task 1 binary segmentation masks will provide enclosing bounding boxes.
- Task 3 CSV diagnoses will provide the target class for each image.
- Class mapping: `0 = melanoma`, `1 = nevus`, `2 = seborrheic_keratosis`.
- Official COCO-pretrained `yolo11n.pt` initializes the model; it is not already
  trained to detect these lesion classes.
- Preserve official split membership for initial preparation. Patient/lesion
  independence still requires an audit and clarification with the tutor. An
  image-level split alone is not evidence of patient independence.
- Do not merge the existing ISIC 2018 course data into these splits: many image
  IDs overlap. The existing 2018 files remain separate.

The assets are intentionally stored **outside this Git checkout**:

```text
3710eassy/
├── PatternAnalysis-2026/recognition/YOLO11_ISIC2017_s4910971/
├── data/ISIC2017/
│   ├── archives/             # original official ZIP downloads
│   ├── raw/                  # extracted original images and masks
│   ├── metadata/             # official diagnosis CSV files
│   └── download_manifest.json
└── weights/yolo11n.pt
```

## Asset preparation

Run from this project directory with Python 3.9 or newer:

```bash
python3 scripts/download_assets.py
python3 scripts/verify_assets.py
```

To run verification only after downloading and extraction succeed:

```bash
python3 scripts/download_assets.py --connections 32 && python3 scripts/verify_assets.py
```

`--connections` controls concurrent HTTP ranges per large archive (default 16,
allowed 1–64). More connections do not guarantee higher throughput. Existing
completed ranges are retained when restarting. Do not run multiple downloaders
against the same workspace simultaneously. Successful full verification writes
`output/setup_verification.json` under the workspace root; a partial download
is not a passed verification.

The downloader resumes partial files, checks response sizes, records SHA256
hashes, verifies the official model SHA256, and checks CRCs during ZIP extraction.
The dataset SHA256 values record provenance; they are not published reference
checksums. Re-running skips completed downloads and verified extraction records.

Use `--workspace-root /absolute/path` to select another workspace, including on
Rangpur. Downloading this full set to the current Rangpur home directory would
exceed the previously reported remaining quota; use an appropriate allocation.

The verifier checks image/mask/diagnosis pairing, expected split counts,
classification values, image decoding, mask dimensions, empty masks and image
ID overlap. It also loads the official model and runs a synthetic-image smoke
test. This smoke test checks execution only, not lesion-detection performance.

## Environment

`requirements.txt` records the already installed local versions used for the
initial setup. Python 3.9.6 on macOS arm64 was detected. Check device availability
in the generated setup verification report; install a suitable PyTorch build
when moving to Rangpur rather than assuming this macOS environment transfers.

## Planned experiments

1. Establish a fine-tuned default-resolution YOLO11n baseline.
2. Compare a higher input resolution while controlling model initialization,
   data splits and other training settings.
3. Select confidence thresholds using validation data only.
4. Evaluate the final choices on the held-out test split; report detection
   quality, class-specific recall, localization IoU, memory and latency.
5. Examine 3–5 actual failures and describe deployment limitations.

The final submission will include `modules.py`, `dataset.py`, `train.py`,
`predict.py` and this report. At this stage `modules.py` only provides the
pretrained-model loading component. Full course requirements are not yet met.

## Artificial Intelligence Usage Disclosure

Codex assisted with planning, repository setup, official asset discovery, and
authoring the asset download/verification scripts and model-loading component.
The student must review the code and verify the results before submission.
Update this section as additional AI-assisted work and student verification occur.
The prior label-matching analysis informed dataset selection; it is not a trained
model result.

## Sources and attribution

- [ISIC official downloads](https://challenge.isic-archive.com/data/#2017)
- [ISIC 2017 diagnosis definitions](https://challenge.isic-archive.com/landing/2017/44/)
- Codella et al., *Skin Lesion Analysis Toward Melanoma Detection: A Challenge at
  the 2017 International Symposium on Biomedical Imaging*,
  [arXiv:1710.05006](https://arxiv.org/abs/1710.05006).
- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Official YOLO11 weights](https://huggingface.co/Ultralytics/YOLO11)

Preserve the downloaded dataset attribution/license files. Ultralytics has its
own software/model licensing; the course repository license does not replace it.
