import os
import webbrowser
from threading import Timer
from flask import Flask, jsonify, send_file, send_from_directory

from ydvt.parser import parse_yolo_dataset
from ydvt.analytics import compute_analytics

app = Flask(__name__)
_dataset_cache = None
_dataset_path = None

def get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = parse_yolo_dataset(_dataset_path)
    return _dataset_cache

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
