import pytest
from ydvt.parser import Dataset, ImageRecord, BBox
from ydvt.analytics import compute_analytics

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
