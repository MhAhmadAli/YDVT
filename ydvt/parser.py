import os
import glob
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

def parse_classes(dataset_path: str) -> Dict[int, str]:
    """Parses classes.txt from the dataset path."""
    classes_path = os.path.join(dataset_path, "classes.txt")
    classes = {}
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
    Assumes .txt files have the same base name as the image files.
    """
    classes = parse_classes(dataset_path)
    
    # Supported image extensions
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    images = []
    
    # Recursively find all images (could be in train/val subfolders)
    for root, _, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in img_exts:
                img_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                txt_path = os.path.join(root, base_name + ".txt")
                
                # Try to get image dimensions
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    width, height = 0, 0
                    print(f"Warning: Could not open image {img_path}: {e}")
                
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
                                    pass # Ignore malformed lines
                images.append(record)
                
    return Dataset(classes=classes, images=images)
