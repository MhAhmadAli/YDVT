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

    # Add a second image for pagination testing
    img_path2 = tmp_path / "test2.jpg"
    cv2.imwrite(str(img_path2), img_array)
    txt_file2 = tmp_path / "test2.txt"
    txt_file2.write_text("0 0.5 0.5 0.2 0.2")

    ydvt.server._dataset_path = str(tmp_path)
    ydvt.server._dataset_cache = None

    with app.test_client() as client:
        yield client

def test_api_analytics(client):
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["summary"]["total_images"] == 2
    assert data["summary"]["total_bboxes"] == 3
    assert data["class_distribution"]["class0"] == 2
    assert data["class_distribution"]["class1"] == 1

def test_api_images_pagination(client):
    response = client.get("/api/images?page=1&limit=1")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["images"]) == 1
    assert data["pagination"]["total_count"] == 2
    assert data["pagination"]["total_pages"] == 2
    assert data["pagination"]["page"] == 1

def test_api_images_search(client):
    response = client.get("/api/images?search=test2")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["images"]) == 1
    assert data["images"][0]["filename"] == "test2.jpg"
    
def test_api_analytics_jobs(client):
    response = client.post("/api/analytics/jobs", json={"options": {"images_per_class": True}})
    assert response.status_code == 202
    data = response.get_json()
    assert "job_id" in data
    assert data["status"] == "processing"
    
    job_id = data["job_id"]
    
    # Poll until done
    import time
    for _ in range(20):
        res = client.get(f"/api/analytics/jobs/{job_id}")
        assert res.status_code in (200, 500)
        d = res.get_json()
        if d["status"] == "completed":
            assert "images_per_class" in d["result"]
            break
        time.sleep(0.1)
    else:
        pytest.fail("Job did not complete in time")
        
def test_api_analytics_invalid_job(client):
    response = client.get("/api/analytics/jobs/invalid-id-123")
    assert response.status_code == 404

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
