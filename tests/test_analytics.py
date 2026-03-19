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


# ---------------------------------------------------------------------------
# Shared fixture: a richer dataset for optional analytics
# ---------------------------------------------------------------------------

def _rich_dataset():
    """3-class dataset with varied bboxes for testing optional analytics."""
    return Dataset(
        classes={0: "cat", 1: "dog", 2: "bird"},
        images=[
            ImageRecord(
                image_path="/ds/train/images/img1.jpg", width=640, height=480,
                bboxes=[
                    BBox(0, 0.5, 0.5, 0.2, 0.2),
                    BBox(1, 0.3, 0.3, 0.1, 0.05),
                ],
            ),
            ImageRecord(
                image_path="/ds/train/images/img2.jpg", width=640, height=480,
                bboxes=[BBox(0, 0.8, 0.8, 0.4, 0.5)],
            ),
            ImageRecord(
                image_path="/ds/valid/images/img3.jpg", width=1920, height=1080,
                bboxes=[BBox(2, 0.1, 0.1, 0.05, 0.05)],
            ),
            ImageRecord(
                image_path="/ds/valid/images/img4.jpg", width=640, height=480,
                bboxes=[],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: Optional analytics
# ---------------------------------------------------------------------------

class TestImagesPerClass:
    def test_counts_unique_images(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"images_per_class": True})
        ipc = r["images_per_class"]
        assert ipc["cat"] == 2   # img1, img2
        assert ipc["dog"] == 1   # img1
        assert ipc["bird"] == 1  # img3


class TestBboxCountPerImage:
    def test_stats(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"bbox_count_per_image": True})
        bpi = r["bbox_count_per_image"]
        assert bpi["min"] == 0   # img4 has 0
        assert bpi["max"] == 2   # img1 has 2
        assert bpi["mean"] == 1.0  # (2+1+1+0)/4
        assert bpi["median"] == 1.0


class TestBboxSizeDist:
    def test_categories(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"bbox_size_dist": True})
        bsd = r["bbox_size_dist"]
        # areas: 0.04, 0.005, 0.20, 0.0025
        # small (<0.01): 0.005, 0.0025 → 2
        # medium (0.01..0.10): 0.04 → 1
        # large (>0.10): 0.20 → 1
        assert bsd["small"] == 2
        assert bsd["medium"] == 1
        assert bsd["large"] == 1


class TestBboxAspectRatio:
    def test_stats(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"bbox_aspect_ratio": True})
        bar = r["bbox_aspect_ratio"]
        # ratios: 0.2/0.2=1.0, 0.1/0.05=2.0, 0.4/0.5=0.8, 0.05/0.05=1.0
        assert bar["min"] == 0.8
        assert bar["max"] == 2.0
        assert "mean" in bar
        assert "median" in bar


class TestLocationHeatmaps:
    def test_grid_shape(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"location_heatmaps": True})
        hm = r["location_heatmaps"]
        assert len(hm) == 10
        assert len(hm[0]) == 10

    def test_total_counts(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"location_heatmaps": True})
        total = sum(sum(row) for row in r["location_heatmaps"])
        assert total == 4  # 4 bboxes total


class TestImageResolutionDist:
    def test_groups_resolutions(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"image_resolution_dist": True})
        ird = r["image_resolution_dist"]
        assert "640x480" in ird
        assert ird["640x480"] == 3  # img1, img2, img4
        assert "1920x1080" in ird
        assert ird["1920x1080"] == 1


class TestLabelDensity:
    def test_avg(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"label_density": True})
        ld = r["label_density"]
        assert ld["total_images"] == 4
        assert ld["total_bboxes"] == 4
        assert ld["avg_bboxes_per_image"] == 1.0


class TestCoOccurrenceMatrix:
    def test_matrix_shape(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"co_occurrence_matrix": True})
        com = r["co_occurrence_matrix"]
        assert "cat" in com
        assert "dog" in com
        assert "bird" in com

    def test_co_occurrence_values(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"co_occurrence_matrix": True})
        com = r["co_occurrence_matrix"]
        # cat and dog co-occur in img1
        assert com["cat"]["dog"] == 1
        assert com["dog"]["cat"] == 1
        # cat appears in 2 images total (self co-occurrence = diagonal)
        assert com["cat"]["cat"] == 2
        # bird never co-occurs with cat or dog
        assert com["bird"]["cat"] == 0


class TestAnnotationCompleteness:
    def test_stats(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"annotation_completeness": True})
        ac = r["annotation_completeness"]
        assert ac["images_with_annotations"] == 3
        assert ac["images_without_annotations"] == 1
        assert ac["percentage_annotated"] == 75.0


class TestDuplicateDetection:
    def test_no_duplicates_synthetic(self):
        """Synthetic paths don't exist on disk, so no hashes → no duplicates."""
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"duplicate_detection": True})
        dd = r["duplicate_detection"]
        assert dd["duplicate_groups"] == 0

    def test_duplicates_found(self, tmp_path):
        import cv2
        import numpy as np
        # Create two identical images
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        p1 = str(tmp_path / "a.jpg")
        p2 = str(tmp_path / "b.jpg")
        cv2.imwrite(p1, img)
        cv2.imwrite(p2, img)
        ds = Dataset(
            classes={0: "x"},
            images=[
                ImageRecord(image_path=p1, width=50, height=50, bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2)]),
                ImageRecord(image_path=p2, width=50, height=50, bboxes=[BBox(0, 0.5, 0.5, 0.2, 0.2)]),
            ],
        )
        r = compute_analytics(ds, options={"duplicate_detection": True})
        dd = r["duplicate_detection"]
        assert dd["duplicate_groups"] == 1
        assert dd["total_duplicate_images"] == 2


class TestLabelImbalance:
    def test_imbalance_stats(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"label_imbalance": True})
        li = r["label_imbalance"]
        # counts: cat=2, dog=1, bird=1
        assert li["max_count"] == 2
        assert li["min_count"] == 1
        assert li["imbalance_ratio"] == 2.0


class TestOutlierDetection:
    def test_detects_outliers(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"outlier_detection": True})
        od = r["outlier_detection"]
        assert "total_outliers" in od
        assert "area_mean" in od
        assert "area_std" in od
        assert isinstance(od["outliers"], list)


class TestAnchorAnalysis:
    def test_anchor_output(self):
        ds = _rich_dataset()
        r = compute_analytics(ds, options={"anchor_analysis": True})
        aa = r["anchor_analysis"]
        assert aa["k"] <= 5
        assert len(aa["anchors"]) > 0
        for a in aa["anchors"]:
            assert "width" in a
            assert "height" in a


class TestDefaultsUnchangedWithoutOptions:
    def test_no_optional_keys_without_options(self):
        ds = _rich_dataset()
        r = compute_analytics(ds)
        # Only default keys should be present
        assert "class_distribution" in r
        assert "split_distribution" in r
        assert "images_per_class" not in r
        assert "co_occurrence_matrix" not in r
        assert "duplicate_detection" not in r
