import pytest
from ydvt.parser import Dataset, ImageRecord, BBox
from ydvt.analytics import compute_analytics, _detect_split

def test_compute_analytics():
    dataset = Dataset(
        classes={0: "cat", 1: "dog"},
        images=[
            ImageRecord(
                image_path="1.jpg", width=100, height=100, 
                bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2), BBox(1, 0.5, 0.5, 0.1, 0.1)]
            ),
            ImageRecord(
                image_path="2.jpg", width=100, height=100, 
                bboxes=[]
            )
        ]
    )

    result = compute_analytics(dataset)

    assert result["summary"]["total_images"] == 2
    assert result["summary"]["images_with_annotations"] == 1
    assert result["summary"]["total_bboxes"] == 2
    
    assert result["class_distribution"]["cat"] == 1
    assert result["class_distribution"]["dog"] == 1
    
    assert result["avg_bbox_sizes"]["cat"]["w"] == pytest.approx(0.2)
    assert result["avg_bbox_sizes"]["cat"]["h"] == pytest.approx(0.2)
    assert result["avg_bbox_sizes"]["dog"]["w"] == pytest.approx(0.1)
    assert result["avg_bbox_sizes"]["dog"]["h"] == pytest.approx(0.1)

def test_unmapped_class():
    dataset = Dataset(
        classes={0: "cat"},
        images=[
            ImageRecord(
                image_path="1.jpg", width=100, height=100, 
                bboxes=[BBox(99, 0.5, 0.5, 0.2, 0.2)]
            )
        ]
    )
    result = compute_analytics(dataset)
    # The analytics module maps missing class IDs as keys directly and tracks them in unmapped_classes
    assert result["summary"]["unmapped_classes"] == [99]
    assert 99 in result["class_distribution"] or "Class 99" in result["class_distribution"]


# ---------------------------------------------------------------------------
# Tests: _detect_split
# ---------------------------------------------------------------------------

class TestDetectSplit:
    def test_train_split(self):
        assert _detect_split("/data/train/images/img1.jpg") == "train"

    def test_valid_split(self):
        assert _detect_split("/data/valid/images/img1.jpg") == "valid"

    def test_val_normalised(self):
        assert _detect_split("/data/val/images/img1.jpg") == "valid"

    def test_validation_normalised(self):
        assert _detect_split("/data/validation/images/img1.jpg") == "valid"

    def test_test_split(self):
        assert _detect_split("/data/test/images/img1.jpg") == "test"

    def test_no_split_returns_unassigned(self):
        assert _detect_split("/data/images/img1.jpg") == "unassigned"

    def test_windows_path(self):
        assert _detect_split("C:\\datasets\\train\\images\\img1.jpg") == "train"

    def test_side_by_side_no_split(self):
        assert _detect_split("/my_dataset/img1.jpg") == "unassigned"


# ---------------------------------------------------------------------------
# Tests: split_distribution in compute_analytics
# ---------------------------------------------------------------------------

class TestSplitDistribution:
    def test_split_distribution_detected(self):
        dataset = Dataset(
            classes={0: "cat"},
            images=[
                ImageRecord(
                    image_path="/ds/train/images/img1.jpg", width=100, height=100,
                    bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2)],
                ),
                ImageRecord(
                    image_path="/ds/train/images/img2.jpg", width=100, height=100,
                    bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2)],
                ),
                ImageRecord(
                    image_path="/ds/valid/images/img3.jpg", width=100, height=100,
                    bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2)],
                ),
            ],
        )
        result = compute_analytics(dataset)
        split = result["split_distribution"]

        assert "train" in split
        assert "valid" in split
        assert split["train"]["images"] == 2
        assert split["valid"]["images"] == 1
        assert split["train"]["percentage"] == pytest.approx(66.7)
        assert split["valid"]["percentage"] == pytest.approx(33.3)
        assert split["train"]["bboxes"] == 2
        assert split["valid"]["bboxes"] == 1

    def test_all_unassigned(self):
        dataset = Dataset(
            classes={0: "cat"},
            images=[
                ImageRecord(
                    image_path="img1.jpg", width=100, height=100,
                    bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2)],
                ),
            ],
        )
        result = compute_analytics(dataset)
        split = result["split_distribution"]
        assert "unassigned" in split
        assert split["unassigned"]["images"] == 1
        assert split["unassigned"]["percentage"] == 100.0
