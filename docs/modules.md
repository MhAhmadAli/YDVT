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

## 4. `ydvt/server.py`
A lightweight `Flask` integration to serve the local dataset GUI.
- Restricts direct File System access securely by only serving images mapped inside the loaded `Dataset` object memory.
- Provides `/api/analytics` and `/api/images` endpoints.

## 5. Web Frontend (`ydvt/templates/*`)
- **HTML**: Standard layout with Sidebar and Data View.
- **CSS**: Premium dark-mode variables using Vanilla CSS. Inter fonts.
- **JS**: Fetches metrics, parses Chart.js config for visualization, and utilizes HTML5 `<canvas>` to accurately plot `(x_center, y_center, width, height)` rectangles onto images recursively scaled to the browser viewport.
