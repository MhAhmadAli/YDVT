# YDVT (YOLO Dataset Visualization Tool)

YDVT is a lightweight, high-performance tool for parsing, analyzing, and interactively visualizing YOLO format image datasets. Built purely in Python, it provides two distinct interfaces without needing a complex decoupled frontend architecture:

1. **Command Line Interface (CLI):** Outputs beautifully formatted terminal summaries of your dataset statistics using `rich`.
2. **Web GUI:** An interactive, dark-themed dashboard delivered straight to your local browser—featuring smooth performance without requiring Node.js or NPM! 

## Features
- **YOLO Parsing:** Deep parsing of YOLO `.txt` files mapping `[class_id, x_center, y_center, width, height]` alongside `classes.txt` indices.
- **Analytics Engine:** Fast metrics computation reporting total instances per class, tracking missing classes, and mapping average bounding box distributions.
- **Embedded Server:** Local lightweight Flask implementation streaming dataset images securely without transferring or duplicating files.
- **Premium Aesthetics:** Web interface generating accurate HTML5 `<canvas>` bounding boxes over datasets and reactive Chart.js metric interfaces natively.

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
*Adding the `--gui` flag will quickly calculate data structures, launch an optimized backend on port 5000, and auto-open a gorgeous visual dashboard natively in your default web browser.*

## Documentation and Testing
- Run test suites safely from the repository root: `pytest tests/` (Tests cover CLI integration, Parser handlers, and Server routes)
- View structure definitions logically detailed inside the `/docs` directory.
