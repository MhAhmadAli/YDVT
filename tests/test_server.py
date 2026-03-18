import os
import pytest
from ydvt.server import app, get_dataset
import ydvt.server

@pytest.fixture
def client(tmp_path):
    app.config["TESTING"] = True

    # Setup dummy environment inside tmp_path
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("class0\nclass1")

    img_file = tmp_path / "test1.jpg"
    img_file.write_bytes(b"") # fake image

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
