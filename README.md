# YDVT (YOLO Dataset Visualization Tool)

YDVT is a lightweight, high-performance tool for parsing, analyzing, and interactively visualizing YOLO format image datasets. Built purely in Python, it provides two distinct interfaces without needing a complex decoupled frontend architecture:

1. **Command Line Interface (CLI):** Outputs beautifully formatted terminal summaries of your dataset statistics using `rich`.
2. **Web GUI:** An interactive, dark-themed dashboard delivered straight to your local browser—featuring smooth performance without requiring Node.js or NPM! 

## Features
- **YOLO Parsing:** Deep parsing of YOLO `.txt` files mapping `[class_id, x_center, y_center, width, height]` alongside `classes.txt` indices. Supports both side-by-side and `images/`/`labels/` directory structures.
- **Analytics Engine:** Fast metrics computation reporting total instances per class, tracking missing classes, and mapping average bounding box distributions.
- **Extended Analytics (CLI):** 13 optional deep-dive metrics enabled via individual flags or `--all-analytics`:
  - Images per class, BBox count per image, BBox size & aspect ratio distributions
  - Object location heatmaps, Image resolution distribution, Label density
  - Class co-occurrence matrix, Annotation completeness, Duplicate image detection
  - Label imbalance metrics, Outlier detection, Anchor box suitability analysis
- **Embedded Server:** Local lightweight Flask implementation streaming dataset images securely without transferring or duplicating files.
- **Premium Aesthetics:** Web interface generating accurate HTML5 `<canvas>` bounding boxes over datasets and reactive Chart.js metric interfaces natively.
- **Data Augmentation:** Per-class augmentation to balance imbalanced datasets. Supports 14 transforms (use these exact keys for the `--augmentations` CLI flag):
  - *Geometric:* `rotate`, `flip_horizontal`, `random_crop`, `resize`, `translate`
  - *Colour:* `brightness`, `contrast`, `saturation`, `hue`
  - *Noise/Regularisation:* `gaussian_blur`, `gaussian_noise`, `cutout`
  - *Multi-image:* `mixup`, `cutmix`
- **Strict Mode:** Optional filter that only uses source images where every bounding box belongs to the target class, preventing non-target class counts from growing.

## Prerequisites
- Python 3.9+
- A valid dataset organized in YOLO format (images matched with their respective `.txt` coordinate files, plus an optional `classes.txt`).

## Installation
Clone the repository, create a virtual environment, and install the tiny footprint dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
Simply point the program to the root directory surrounding your YOLO dataset.

**1. Fast Terminal Analytics (CLI)**
```bash
python3 ydvt /path/to/my/dataset
```
*(Alternatively: `python3 -m ydvt.main /path/to/my/dataset`)*

**2. Interactive Visual Dashboard (GUI)**
```bash
python3 ydvt /path/to/my/dataset --gui
```
*Adding the `--gui` flag will launch an optimized backend on port 5000 and auto-open a visual dashboard in your default browser. The dashboard includes an **✦ Augment** button to open the augmentation modal where you can select target classes, choose augmentations, enable Strict Mode, and generate balanced training data.*

**3. Interactive Augmentation Wizard (CLI)**
```bash
python3 ydvt /path/to/my/dataset --augment
```
*Launches a step-by-step terminal wizard that guides you through:*
1. *Selecting target classes (with instance counts displayed)*
2. *Choosing augmentations to apply*
3. *Setting the number of images to generate per class*
4. *Enabling optional Strict Mode*
5. *Confirming and executing the augmentation*

**4. Headless Augmentation (CLI)**
```bash
python3 ydvt /path/to/my/dataset --augment --classes dog cat --augmentations rotate mixup --num-images 10 --strict-mode
```
*Bypass the interactive wizard and run programmatic augmentations. Requires both `--classes` and `--augmentations`. Parameters:*
- `--classes`: Target classes to augment (names or numeric IDs)
- `--augmentations`: Augmentations to apply (e.g., rotate, mixup)
- `--num-images`: Number of images to generate per class (default: 5)
- `--strict-mode`: Enable strict mode filtering

**5. Extended Analytics (CLI)**
```bash
# Enable individual metrics
python3 ydvt /path/to/my/dataset --co-occurrence-matrix --label-imbalance

# Enable all optional analytics at once
python3 ydvt /path/to/my/dataset --all-analytics
```
*Available flags: `--images-per-class`, `--bbox-count-per-image`, `--bbox-size-dist`, `--bbox-aspect-ratio`, `--location-heatmaps`, `--image-resolution-dist`, `--label-density`, `--co-occurrence-matrix`, `--annotation-completeness`, `--duplicate-detection`, `--label-imbalance`, `--outlier-detection`, `--anchor-analysis`*

## Documentation and Testing
- Run test suites from the repository root: `pytest tests/` (86 tests covering parser, analytics, augmenter, server routes, headless runner, and CLI wizard)
- View module documentation inside the `/docs` directory.
