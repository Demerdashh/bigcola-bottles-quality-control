# Dataset Card - Big Cola Bottles Quality Control

## Dataset Summary

A manually created image dataset for industrial bottle quality control, built to train a real-time object detection model on a simulated factory conveyor belt environment. Every image was physically staged, captured, and annotated by the author. No existing dataset or synthetic generation was used.

---

## Dataset Details

| Field | Value |
|---|---|
| **Total images** | 376 |
| **Number of classes** | 5 |
| **Annotation format** | YOLO (normalized `x_center y_center width height`) |
| **Image resolution** | 3000x4000 |
| **Capture device** | Samsung s24 Ultra |
| **Annotation tool** | CVAT |
| **Dataset version** | 1.0 |
| **Author** | Youssef El Demerdash |

---

## Classes

| ID | Name | Description |
|---|---|---|
| 0 | `Perfect_bottle` | Bottle with cap, label, correct fill level - no defects |
| 1 | `empty_bottle` | Bottle with no liquid content |
| 2 | `no_cap` | Bottle missing its cap |
| 3 | `no_label` | Bottle missing its label |
| 4 | `crooked_cap` | Bottle with a misaligned or improperly sealed cap |

---

## Class Distribution (Before Augmentation)

| Class | Raw Image Count |
|---|---|
| Perfect_bottle | 193 |
| empty_bottle | 118 |
| no_cap | 113 |
| no_label | 95 |
| crooked_cap | 90 |

> Class 3 (`no_label`) was the most unrecognized class by the model and received an additional 50 augmentation samples on top of the standard balancing target.

---

## Data Collection Methodology

### Physical Setup

The dataset was captured in a controlled environment designed to simulate industrial conveyor belt conditions:

- **Background:** (randomized desktop wallpapers and some images had large black paper used as the conveyor surface) - chosen to maximise contrast against the bottle labels and provide clean, consistent background conditions across all images
- **Stability:** A dedicated tripod was used for every capture session to ensure consistent camera angle, height, and distance across all 376 images
- **Lighting:** indoor artificial light / ring light
- **Camera distance:** 20~30 cm to bottle surface
- **Bottle positioning:** Bottles were repositioned manually between shots to introduce natural variation in horizontal placement, slight rotation, and perspective

### Image Capture Process

1. Physical Big Cola bottles were purchased specifically for this project
2. Each bottle condition was staged manually (cap removed, label removed, cap intentionally misaligned, etc.)
3. Multiple images were captured per condition with varied positioning
4. Images were transferred to Google Drive for annotation and pipeline processing

---

## Annotation Process

All 376 images were labeled manually using CVAT labeling tool. Each bounding box was drawn to tightly encompass the visible bottle, with class assigned based on the bottle's condition at the time of capture.

**Label format:** YOLO `.txt` files — one file per image, one line per bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to `[0, 1]` relative to image dimensions.

**Label quality note:** Some annotation sessions produced class IDs as floats (e.g., `4.0` instead of `4`). This was detected during pipeline validation and normalized using `int(float(class_id))` parsing. See `docs/engineering_debrief.md` — Bug 2.

---

## Augmentation Strategy

To address class imbalance and increase dataset size for training, Albumentations was used to generate synthetic samples up to a target of **250 images per class**.

Transforms applied:
- `HorizontalFlip`
- `RandomBrightnessContrast`
- `HueSaturationValue`
- `Rotate`
- `Affine` (conservative `translate_percent ≤ 0.03` — see engineering debrief Bug 4)
- `GaussianBlur`
- `CLAHE`

All augmented bounding boxes were validated post-generation to confirm coordinates remain within `[0, 1]` and dimensions are positive.

---

## Dataset Integrity

The raw label directory is fingerprinted with SHA-256 at the start of every pipeline run. This hash is stored in `model_card.json` to guarantee that the exact same data was used to produce the published model weights.

SHA-256 prefix: `26a0b22513643732`

---

## Train / Validation Split

| Split | Images |
|---|---|
| Train | 907  (319 unique scenes) |
| Validation | 160  (57 unique scenes) |

Split ratio: **85% train / 15% validation**

Split strategy: Grouped by original image stem. All augmented versions of a source image are always assigned to the same split as their source - this prevents data leakage where the model sees augmented variants of a validation image during training.

---

## Limitations

- Dataset was captured in a single controlled indoor environment. Real factory conditions (varying lighting, motion blur, multiple bottles simultaneously on belt) may require additional fine-tuning data
- 376 raw images is a constrained dataset size - augmentation partially compensates but cannot fully substitute for real distribution diversity
- All images contain a single bottle in frame. Multi-bottle detection (realistic conveyor throughput) is outside the scope of this dataset version

---

## How to Access

| Platform | Link |
|---|---|
| Kaggle | https://www.kaggle.com/datasets/youssefeldemerdashh/big-cola-bottles-dataset-for-yolo-quality-control |

---

*Author: Youssef El Demerdash*  
*Project repository: https://github.com/Demerdashh/bigcola-bottles-quality-control*
