import os
import webbrowser
from threading import Timer
from flask import Flask, jsonify, request, send_file, send_from_directory

from ydvt.parser import parse_yolo_dataset
from ydvt.analytics import compute_analytics
from ydvt.augmenter import apply_augmentations, list_available_augmentations

app = Flask(__name__)
_dataset_cache = None
_dataset_path = None

def get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = parse_yolo_dataset(_dataset_path)
    return _dataset_cache

def invalidate_cache():
    """Clear the dataset cache so the next request re-parses from disk."""
    global _dataset_cache
    _dataset_cache = None

@app.route("/")
def index():
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    return send_from_directory(templates_dir, "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    return send_from_directory(templates_dir, filename)

@app.route("/api/analytics")
def api_analytics():
    ds = get_dataset()
    data = compute_analytics(ds)
    return jsonify(data)

@app.route("/api/images")
def api_images():
    ds = get_dataset()
    images_data = []
    for i, record in enumerate(ds.images):
        images_data.append({
            "idx": i,
            "filename": os.path.basename(record.image_path),
            "width": record.width,
            "height": record.height,
            "bboxes": [{"class_id": b.class_id, "x_center": b.x_center, "y_center": b.y_center, 
                        "width": b.width, "height": b.height} for b in record.bboxes]
        })
    return jsonify({"images": images_data, "classes": ds.classes})

@app.route("/api/image/<int:idx>")
def api_image_file(idx):
    ds = get_dataset()
    if 0 <= idx < len(ds.images):
        return send_file(ds.images[idx].image_path)
    return "Image not found", 404

@app.route("/api/augmentations")
def api_augmentations():
    """Return the list of available augmentations."""
    return jsonify(list_available_augmentations())

@app.route("/api/augment", methods=["POST"])
def api_augment():
    """
    Apply augmentations to balance specified classes.

    Expects JSON body::

        {
            "target_classes": [0, 2],
            "augmentations": ["rotate", "flip_horizontal", "gaussian_blur"],
            "num_images": 5,
            "strict_filter": false,  // optional, default false
            "params": {}             // optional per-augmentation overrides
        }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    target_classes = data.get("target_classes")
    augmentation_names = data.get("augmentations")
    num_images = data.get("num_images", 5)
    strict_filter = data.get("strict_filter", False)
    params = data.get("params", {})

    if not target_classes or not augmentation_names:
        return jsonify({"error": "target_classes and augmentations are required"}), 400

    ds = get_dataset()

    try:
        result = apply_augmentations(
            dataset=ds,
            target_classes=target_classes,
            augmentation_names=augmentation_names,
            num_images=num_images,
            params=params,
            strict_filter=strict_filter,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Invalidate the cache so subsequent analytics/image requests reflect the new data
    invalidate_cache()

    return jsonify(result)

def start_server(dataset_path: str, port: int = 5000):
    global _dataset_path
    _dataset_path = os.path.abspath(dataset_path)
    
    def open_browser():
        webbrowser.open_new(f"http://127.0.0.1:{port}/")
        
    Timer(1.0, open_browser).start()
    
    print(f"Starting YDVT GUI server on http://127.0.0.1:{port}/\nServing dataset from {_dataset_path}")
    print("Press CTRL+C to quit")
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host="127.0.0.1", port=port, debug=False)
