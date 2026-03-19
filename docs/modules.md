# YDVT Modules Documentation

This covers the core components of the Image Dataset Visualization Tool.

## 1. `ydvt/parser.py`
Responsible for reading the local Dataset format (currently YOLO).
- `parse_yolo_dataset(dataset_path: str) -> Dataset`: Scans the directory recursively finding images and matching `.txt` YOLO annotation files alongside `classes.txt`. Wraps everything in a structured layout.

## 2. `ydvt/analytics.py`
Computes the statistics required for visual reporting.
- `compute_analytics(dataset: Dataset) -> Dict`: Iterates over the records to find total bounding boxes, missing classes, class frequency distribution, and computes average relative sizes `(width, height)`.

## 3. `ydvt/main.py`
The CLI Entrypoint.
- Resolves CLI Arguments via `argparse`.
- Computes logic and uses `rich` Table/Panel formatting to print a beautiful output summary to terminal.
- Routes execution to `server.py` if the `--gui` flag is active.
- Routes execution to `wizard.py` if the `--augment` flag is active.

## 4. `ydvt/wizard.py`
Interactive CLI augmentation wizard.
- Uses `questionary` for multi-select checkbox prompts (class selection, augmentation selection).
- Presents class distribution counts to help identify underrepresented classes.
- Validates user input (image count) and shows a confirmation summary before execution.
- Uses `rich` for styled output and a progress spinner during augmentation.

## 5. `ydvt/server.py`
A lightweight `Flask` integration to serve the local dataset GUI.
- Restricts direct File System access securely by only serving images mapped inside the loaded `Dataset` object memory.
- Provides `/api/analytics` and `/api/images` endpoints.
- **`GET /api/augmentations`**: Returns the list of available augmentation transforms with metadata.
- **`POST /api/augment`**: Accepts a JSON body with `target_classes`, `augmentations`, `num_images`, and optional `params`. Runs the augmentation pipeline, saves new images/labels, and invalidates the dataset cache.

## 6. `ydvt/augmenter.py`
Data augmentation module for balancing imbalanced YOLO datasets.
- Uses `albumentations` for bbox-aware image transforms.
- **Standard augmentations** (12): Rotate, Horizontal Flip, Random Crop, Resize/Scale, Translate, Brightness, Contrast, Saturation, Hue, Gaussian Blur, Gaussian Noise, Cutout/Random Erasing.
- **Advanced augmentations** (2): Mixup (alpha-blend two images) and CutMix (paste a crop of one image onto another), with merged bounding boxes.
- `apply_augmentations(dataset, target_classes, augmentation_names, num_images, params)`: Main entry point. For each target class, samples source images, applies the selected pipeline, and writes augmented images + YOLO labels to disk.
- `list_available_augmentations()`: Returns metadata for all 14 augmentations for UI rendering.
- Respects both side-by-side and `images/`/`labels/` directory structures.

## 7. Web Frontend (`ydvt/templates/*`)
- **HTML**: Standard layout with Sidebar and Data View. Includes an Augmentation modal dialog.
- **CSS**: Premium dark-mode variables using Vanilla CSS. Inter fonts. Modal overlay with animations.
- **JS**: Fetches metrics, parses Chart.js config for visualization, and utilizes HTML5 `<canvas>` to accurately plot `(x_center, y_center, width, height)` rectangles onto images recursively scaled to the browser viewport. Augmentation panel populates class distribution and available transforms, sends configuration to `/api/augment`, and refreshes charts on completion.

