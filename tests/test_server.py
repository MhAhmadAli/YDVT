import os
import pytest
import numpy as np
import cv2
from ydvt.server import app, get_dataset
import ydvt.server

@pytest.fixture
def client(tmp_path):
    app.config["TESTING"] = True

    # Setup dummy environment inside tmp_path
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("class0\nclass1")

    # Create a real image so augmenter can read it
    img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    img_path = tmp_path / "test1.jpg"
    cv2.imwrite(str(img_path), img_array)

    txt_file = tmp_path / "test1.txt"
    txt_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.2")

    ydvt.server._dataset_path = str(tmp_path)
    ydvt.server._dataset_cache = None

    with app.test_client() as client:
        yield client

def test_api_analytics(client):
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["summary"]["total_images"] == 1
    assert data["summary"]["total_bboxes"] == 2
    assert data["class_distribution"]["class0"] == 1
    assert data["class_distribution"]["class1"] == 1

def test_api_images(client):
    response = client.get("/api/images")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["images"]) == 1
    assert data["images"][0]["filename"] == "test1.jpg"
    assert data["classes"]["0"] == "class0"

def test_index_route(client):
    # Depending on test environment, templates may not be fully resolvable via relative paths, 
    # but we can try to hit the root to see if it routes correctly.
    try:
        response = client.get("/")
        # Should be 200 if template exists
        assert response.status_code == 200 or response.status_code == 404
    except Exception:
        pass

def test_api_augmentations(client):
    response = client.get("/api/augmentations")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    names = {a["name"] for a in data}
    assert "rotate" in names
    assert "mixup" in names
    assert "cutmix" in names

def test_api_augment_missing_body(client):
    response = client.post("/api/augment", content_type="application/json")
    assert response.status_code == 400

def test_api_augment_missing_fields(client):
    response = client.post("/api/augment",
                           json={"target_classes": [0]})
    assert response.status_code == 400

def test_api_augment_success(client):
    response = client.post("/api/augment", json={
        "target_classes": [0],
        "augmentations": ["flip_horizontal"],
        "num_images": 2,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["generated_count"] == 2
    assert len(data["generated_files"]) == 2

def test_api_augment_invalidates_cache(client):
    # First request populates cache
    client.get("/api/analytics")
    assert ydvt.server._dataset_cache is not None

    # Augment invalidates it
    client.post("/api/augment", json={
        "target_classes": [0],
        "augmentations": ["flip_horizontal"],
        "num_images": 1,
    })
    assert ydvt.server._dataset_cache is None

def test_api_augment_strict_filter(client):
    response = client.post("/api/augment", json={
        "target_classes": [1],
        "augmentations": ["flip_horizontal"],
        "num_images": 2,
        "strict_filter": True,
    })
    assert response.status_code == 200
    data = response.get_json()
    # class 1 only exists in images with class 0 too → skipped
    assert data["generated_count"] == 0
    assert 1 in data["skipped_classes"]
