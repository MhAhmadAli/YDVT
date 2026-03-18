import os
import pytest
import yaml
from ydvt.parser import parse_yolo_dataset, BBox

def test_parse_yolo_dataset(tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("class0\nclass1")

    img_file = tmp_path / "test1.jpg"
    img_file.write_bytes(b"") # fake image

    txt_file = tmp_path / "test1.txt"
    txt_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1")

    dataset = parse_yolo_dataset(str(tmp_path))

    assert dataset.classes[0] == "class0"
    assert dataset.classes[1] == "class1"
    assert len(dataset.images) == 1
    
    bboxes = dataset.images[0].bboxes
    assert len(bboxes) == 2
    assert bboxes[0] == BBox(0, 0.5, 0.5, 0.2, 0.2)
    assert bboxes[1] == BBox(1, 0.3, 0.3, 0.1, 0.1)

def test_malformed_annotations(tmp_path):
    img_file = tmp_path / "test1.jpg"
    img_file.write_bytes(b"")

    txt_file = tmp_path / "test1.txt"
    txt_file.write_text("0 0.5 0.5 0.2 0.2\nweird line")

    dataset = parse_yolo_dataset(str(tmp_path))
    assert len(dataset.images) == 1
    assert len(dataset.images[0].bboxes) == 1

def test_parse_yolo_dataset_with_yaml_list(tmp_path):
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("names:\n  - cat\n  - dog\n")
    
    dataset = parse_yolo_dataset(str(tmp_path))
    assert dataset.classes[0] == "cat"
    assert dataset.classes[1] == "dog"

def test_parse_yolo_dataset_with_yaml_dict(tmp_path):
    yaml_file = tmp_path / "dataset.yaml"
    yaml_file.write_text("names:\n  0: apple\n  1: orange\n")
    
    dataset = parse_yolo_dataset(str(tmp_path))
    assert dataset.classes[0] == "apple"
    assert dataset.classes[1] == "orange"

def test_parse_yolo_dataset_separated_structure(tmp_path):
    # Simulated structure: dataset/train/images and dataset/train/labels
    train_dir = tmp_path / "train"
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"
    
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    
    img_file = images_dir / "test2_hash.jpg"
    img_file.write_bytes(b"")
    
    txt_file = labels_dir / "test2_hash.txt"
    txt_file.write_text("2 0.1 0.1 0.5 0.5")
    
    dataset = parse_yolo_dataset(str(tmp_path))
    assert len(dataset.images) == 1
    assert dataset.images[0].image_path == str(img_file)
    assert len(dataset.images[0].bboxes) == 1
    assert dataset.images[0].bboxes[0] == BBox(2, 0.1, 0.1, 0.5, 0.5)
