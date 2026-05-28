# Engineering Debrief — Big Cola Bottle Quality Control Pipeline

## Overview

This document records the six critical bugs encountered and resolved during the development of the Big Cola quality-control CV pipeline. Each bug is documented with its root cause, how it was diagnosed, and the architectural fix applied.

---

## Bug 1 — Double-Transformation of Bounding Box Coordinates

**Stage:** Augmentation  
**Symptom:** Bounding boxes visually drifted or landed outside image boundaries after augmentation. Augmented label files contained coordinates that no longer matched the visible bottle positions.

**Root Cause:** Bounding box coordinates were being manually converted from YOLO normalized format before being passed to Albumentations, and then Albumentations was applying its own YOLO-format coordinate handling on top — effectively transforming the coordinates twice.

**Why It's Subtle:** YOLO format uses `[x_center, y_center, width, height]` all normalized to `[0, 1]`. Albumentations handles this correctly internally when you declare `format='yolo'`. Any manual pre-processing of the coordinates before passing them in causes double application of the transform math, which looks correct in code but produces systematically wrong output.

**Fix:** Never manually convert or normalize YOLO coordinates before passing to Albumentations when `format='yolo'` is declared. Pass raw YOLO values directly. The library handles the coordinate space internally.

```python
# WRONG — pre-converting before passing to Albumentations
coords = [x * img_w, y * img_h, w * img_w, h * img_h]

# CORRECT — pass YOLO values directly
bboxes = [[x_c, y_c, w, h, class_id]]  # raw normalized values
transform = A.Compose([...], bbox_params=A.BboxParams(format='yolo'))
```

---

## Bug 2 — Float Class IDs Breaking Label Parsing

**Stage:** Dataset validation  
**Symptom:** Hard `ValueError` crash during label parsing on a subset of images. Not all labels failed - only those from a specific annotation session.

**Root Cause:** Some annotation tools write class IDs as floats (`4.0`, `0.0`) instead of integers (`4`, `0`). Python's `int()` cannot parse the string `'4.0'` — it raises `ValueError: invalid literal for int()`. The problem was invisible in the raw files and only surfaced during parsing.

**Why It's Subtle:** The labels are visually correct when you open the `.txt` file. `4.0` and `4` represent the same class. The bug is entirely a string-parsing edge case, not a labeling error.

**Fix:** Use `int(float(parts[0]))` instead of `int(parts[0])` when parsing class IDs from YOLO label files.

```python
# WRONG — crashes on '4.0'
class_id = int(parts[0])

# CORRECT — handles both '4' and '4.0'
class_id = int(float(parts[0]))
```

**Lesson:** Never to assume that the label files have a single consistent format, even within my own dataset. Different annotation sessions can produce different string representations of the same value using the same tool.

---

## Bug 3 — Stale Augmented Files Corrupting Runs

**Stage:** Augmentation output  
**Symptom:** After modifying augmentation parameters and re-running the pipeline, the new training run contained a mix of images from the old and new augmentation configurations. Results were unreproducible across runs with identical settings.

**Root Cause:** The `OUTPUT_DIR` was not wiped before each augmentation run. New files were written into the directory, but old files from the previous run that were no longer being generated (e.g., from a class that was previously over-augmented) remained on disk and were picked up by the training data loader.

**Fix:** Wipe `OUTPUT_DIR` completely at the start of every augmentation run before writing any new files.

```python
import shutil
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)
```

**Lesson:** Any pipeline stage that writes files to a directory must own that directory completely, partial overwrites produce silent data contamination that is very difficult to detect downstream (It was the hardest error to solve).

---

## Bug 4 — Negative Bounding Box Widths from Aggressive Affine Transforms

**Stage:** Augmentation  
**Symptom:** Post-augmentation validity scan flagged labels with `width <= 0` or `height <= 0`. These files would silently corrupt training by providing a malformed supervision signal to the model.

**Root Cause:** The Albumentations `Affine` transform with aggressive `translate_percent` values was pushing bounding boxes partially or fully outside the image boundary. When the coordinates were clamped to `[0, 1]`, the resulting width or height could become zero or negative — a geometrically invalid bounding box that YOLO cannot process.

**Fix:** Keep `translate_percent` conservative (≤ 0.03). Add a post-augmentation validity scan that checks all generated label files for coordinates outside `[0, 1]` and bounding box dimensions ≤ 0, and discards any invalid samples before training.

```python
# Conservative — safe
translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}

# Aggressive — causes coordinate clamping artifacts
translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
```

---

## Bug 5 — Visualizer Showing Duplicate Images via Fallback Mechanism

**Stage:** Dataset visualization / QA  
**Symptom:** The visual sanity-check grid appeared to show correctly labeled images, but several grid cells contained the same image repeated. The QA pass was effectively checking the same sample multiple times and providing false confidence.

**Root Cause:** The visualizer had a fallback mechanism that reused source images when it could not find a corresponding augmented sample (There was an issue in the stem_to_imgs function). The fallback silently substituted already-seen images without any warning, making the QA grid look populated while hiding gaps in the augmented output.

**Fix:** Remove all fallback mechanisms from the visualizer. If a sample cannot be found, the grid cell should either be left blank or raise an explicit warning. Silent substitution of any kind invalidates the purpose of a QA pass.

---

## Bug 6 — `cv2.cvtColor` Crash from Stale File Path Dictionary

**Stage:** Visualization / inference  
**Symptom:** Intermittent `cv2.error` crash during image loading. The crash was not reproducible on every run — it depended on execution order.

**Root Cause:** A dictionary mapping image stems to file paths was built once at the start of the session. Later in the pipeline, files were deleted or moved (OUTPUT_DIR wipe, augmentation re-run), but the dictionary still held the old paths. When `cv2.imread()` was called using a stale path, it returned `None`, and the subsequent `cv2.cvtColor(None, ...)` call raised a hard crash.

**Fix:** Never use a pre-built path dictionary across pipeline stages that modify the filesystem. Always resolve paths dynamically at the point of use, or rebuild the dictionary immediately before any read operation.

```python
# WRONG — dictionary built once, used after files may have moved
image_map = {p.stem: p for p in OUTPUT_DIR.glob('*.jpg')}
# ... many cells later, files deleted, map is stale ...
img = cv2.imread(str(image_map[stem]))  # may be None

# CORRECT — resolve path fresh at point of use
img_path = OUTPUT_DIR / 'images' / f'{stem}.jpg'
if not img_path.exists():
    continue
img = cv2.imread(str(img_path))
```

---

## Summary Table

| # | Bug | Stage | Detection Method | Severity |
|---|---|---|---|---|
| 1 | Double coordinate transformation | Augmentation | Visual bbox inspection | High |
| 2 | Float class ID parsing failure | Validation | Hard crash (ValueError) | High |
| 3 | Stale augmented files in OUTPUT_DIR | Augmentation | Unreproducible results across runs | High |
| 4 | Negative bbox dimensions from aggressive Affine | Augmentation | Post-augmentation validity scan | Medium |
| 5 | Visualizer fallback duplicating images | QA / Visualization | Manual grid inspection | Medium |
| 6 | Stale path dictionary causing cv2 crash | Visualization | Intermittent crash (cv2.error) | Medium |

---

## Architectural Principles Applied After Debugging

These six bugs led directly to the following design rules that are now encoded into the pipeline architecture:

1. **Idempotency** — every run produces the same output regardless of previous state (OUTPUT_DIR wipe enforces this)
2. **No silent failures** — every validation step either passes with a confirmation message or raises a hard assertion, never continues quietly with bad data
3. **Reproducibility** — SHA-256 dataset fingerprinting ensures the exact same data goes into every training run
4. **Dynamic path resolution** — file paths are resolved fresh at the point of use, never cached across stages that modify the filesystem
5. **Separation of raw and derived data** — the original labeled dataset is never modified by the pipeline; all transforms operate on copies

---

*Author: Youssef El Demerdash*
