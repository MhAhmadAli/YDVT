import os
import glob
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from PIL import Image

@dataclass
class BBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

@dataclass
class ImageRecord:
    image_path: str
    width: int
    height: int
    bboxes: List[BBox] = field(default_factory=list)

@dataclass
class Dataset:
    classes: Dict[int, str]
    images: List[ImageRecord]

def get_dataset_classes(dataset_path: str) -> Dict[int, str]:
    """
    Attempts to parse class names from YOLO yaml configs (e.g. data.yaml, dataset.yaml)
    or falls back to classes.txt.
    """
    classes = {}
    
    yaml_files = glob.glob(os.path.join(dataset_path, "*.yaml"))
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "names" in data:
                    names = data["names"]
                    if isinstance(names, list):
                        return {idx: name for idx, name in enumerate(names)}
                    elif isinstance(names, dict):
                        return {int(idx): name for idx, name in names.items()}
        except Exception as e:
            print(f"Warning: Failed to parse YAML {yf}: {e}")
            
    classes_path = os.path.join(dataset_path, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    classes[idx] = line
                    
    return classes

def parse_yolo_dataset(dataset_path: str) -> Dataset:
    """
    Parses a directory containing YOLO format images and text annotations.
    Checks for labels placed side-by-side with images or in a sibling `labels/` directory.
    """
    classes = get_dataset_classes(dataset_path)
    
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in img_exts:
                img_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                
                # 1. Check side-by-side
                txt_path = os.path.join(root, base_name + ".txt")
                
                # 2. If not side-by-side, check sibling 'labels' directory
                if not os.path.exists(txt_path):
                    if os.path.basename(root) == 'images':
                        parent_dir = os.path.dirname(root)
                        alt_txt_path = os.path.join(parent_dir, 'labels', base_name + ".txt")
                        if os.path.exists(alt_txt_path):
                            txt_path = alt_txt_path
                
                width, height = 0, 0
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    pass
                
                record = ImageRecord(image_path=img_path, width=width, height=height)
                
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    bbox_width = float(parts[3])
                                    bbox_height = float(parts[4])
                                    
                                    record.bboxes.append(BBox(
                                        class_id=class_id,
                                        x_center=x_center,
                                        y_center=y_center,
                                        width=bbox_width,
                                        height=bbox_height
                                    ))
                                except ValueError:
                                    pass
                images.append(record)
                
    return Dataset(classes=classes, images=images)
