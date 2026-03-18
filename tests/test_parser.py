import pytest
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
    
    # Just one image but two bboxes
    bboxes = dataset.images[0].bboxes
    assert len(bboxes) == 2
    assert bboxes[0] == BBox(0, 0.5, 0.5, 0.2, 0.2)
    assert bboxes[1] == BBox(1, 0.3, 0.3, 0.1, 0.1)

def test_malformed_annotations(tmp_path):
    img_file = tmp_path / "test1.jpg"
    img_file.write_bytes(b"")

    txt_file = tmp_path / "test1.txt"
    # One good, one bad line
    txt_file.write_text("0 0.5 0.5 0.2 0.2\nweird line")

    dataset = parse_yolo_dataset(str(tmp_path))
    assert len(dataset.images) == 1
    assert len(dataset.images[0].bboxes) == 1
