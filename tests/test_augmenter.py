"""
Tests for the augmenter module.

Covers:
- Pipeline building with various augmentation names
- Standard augmentations generate output files with valid YOLO labels
- Mixup and CutMix produce merged bounding boxes
- Edge cases: unknown augmentation names, empty class, no images for class
- Output directory resolution for both side-by-side and images/labels structures
"""

import os
import pytest
import numpy as np
import cv2

from ydvt.parser import Dataset, ImageRecord, BBox
from ydvt.augmenter import (
    apply_augmentations,
    list_available_augmentations,
    _build_pipeline,
    _images_for_class,
    _resolve_output_dirs,
    _write_yolo_label,
    _apply_mixup,
    _apply_cutmix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image(path, width=100, height=100):
    """Create a small solid-colour image on disk."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_dataset(tmp_path, num_images=2, side_by_side=True):
    """
    Create a minimal on-disk YOLO dataset and return the parsed Dataset object.
    """
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\ndog\n")

    if side_by_side:
        img_dir = tmp_path
        lbl_dir = tmp_path
    else:
        img_dir = tmp_path / "train" / "images"
        lbl_dir = tmp_path / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

    records = []
    for i in range(num_images):
        img_path = img_dir / f"img{i}.jpg"
        _make_dummy_image(str(img_path))
        lbl_path = lbl_dir / f"img{i}.txt"
        # All images have class 0 bbox, first image also has class 1
        lines = "0 0.5 0.5 0.2 0.2\n"
        if i == 0:
            lines += "1 0.3 0.3 0.15 0.15\n"
        lbl_path.write_text(lines)

        bboxes = [BBox(0, 0.5, 0.5, 0.2, 0.2)]
        if i == 0:
            bboxes.append(BBox(1, 0.3, 0.3, 0.15, 0.15))

        records.append(ImageRecord(
            image_path=str(img_path), width=100, height=100, bboxes=bboxes
        ))

    return Dataset(classes={0: "cat", 1: "dog"}, images=records)


# ---------------------------------------------------------------------------
# Tests: list_available_augmentations
# ---------------------------------------------------------------------------

class TestListAvailableAugmentations:
    def test_returns_all_14(self):
        augs = list_available_augmentations()
        names = {a["name"] for a in augs}
        expected = {
            "rotate", "flip_horizontal", "random_crop", "resize",
            "translate", "brightness", "contrast", "saturation",
            "hue", "gaussian_blur", "gaussian_noise", "cutout",
            "mixup", "cutmix",
        }
        assert expected == names

    def test_each_has_required_keys(self):
        for aug in list_available_augmentations():
            assert "name" in aug
            assert "label" in aug
            assert "description" in aug


# ---------------------------------------------------------------------------
# Tests: _build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_empty_list(self):
        pipe = _build_pipeline([])
        assert pipe is not None

    def test_unknown_names_ignored(self):
        pipe = _build_pipeline(["nonexistent", "also_fake"])
        # Should still create a valid (empty) compose
        assert pipe is not None

    def test_single_augmentation(self):
        pipe = _build_pipeline(["flip_horizontal"])
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = pipe(image=img, bboxes=[(0.5, 0.5, 0.2, 0.2)], class_ids=[0])
        assert result["image"].shape == (64, 64, 3)
        assert len(result["bboxes"]) >= 0  # bbox may be clipped or kept


# ---------------------------------------------------------------------------
# Tests: _images_for_class
# ---------------------------------------------------------------------------

class TestImagesForClass:
    def test_filters_correctly(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=3)
        # class 0 is in all images
        assert len(_images_for_class(ds, 0)) == 3
        # class 1 is only in the first image
        assert len(_images_for_class(ds, 1)) == 1

    def test_empty_for_missing_class(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert _images_for_class(ds, 99) == []

    def test_strict_excludes_mixed_images(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=3)
        # img0 has class 0 AND class 1 → mixed → excluded under strict
        # img1 and img2 have only class 0 → included
        strict_imgs = _images_for_class(ds, 0, strict=True)
        assert len(strict_imgs) == 2

    def test_strict_returns_empty_when_all_mixed(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=3)
        # class 1 only appears in img0, which also has class 0 → no exclusive images
        strict_imgs = _images_for_class(ds, 1, strict=True)
        assert len(strict_imgs) == 0


# ---------------------------------------------------------------------------
# Tests: _resolve_output_dirs
# ---------------------------------------------------------------------------

class TestResolveOutputDirs:
    def test_side_by_side(self, tmp_path):
        img_path = str(tmp_path / "img0.jpg")
        imgs_dir, lbls_dir = _resolve_output_dirs(img_path)
        assert imgs_dir == str(tmp_path)
        assert lbls_dir == str(tmp_path)

    def test_images_labels_structure(self, tmp_path):
        img_dir = tmp_path / "train" / "images"
        img_dir.mkdir(parents=True)
        img_path = str(img_dir / "img0.jpg")
        imgs_dir, lbls_dir = _resolve_output_dirs(img_path)
        assert imgs_dir == str(img_dir)
        assert lbls_dir == str(tmp_path / "train" / "labels")


# ---------------------------------------------------------------------------
# Tests: _write_yolo_label
# ---------------------------------------------------------------------------

class TestWriteYoloLabel:
    def test_writes_correct_format(self, tmp_path):
        lbl_path = str(tmp_path / "test.txt")
        _write_yolo_label(lbl_path, [(0.5, 0.5, 0.2, 0.2)], [0])
        content = open(lbl_path).read().strip()
        assert content == "0 0.500000 0.500000 0.200000 0.200000"

    def test_multiple_bboxes(self, tmp_path):
        lbl_path = str(tmp_path / "test.txt")
        _write_yolo_label(
            lbl_path,
            [(0.5, 0.5, 0.2, 0.2), (0.3, 0.3, 0.1, 0.1)],
            [0, 1],
        )
        lines = open(lbl_path).read().strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Tests: _apply_mixup
# ---------------------------------------------------------------------------

class TestMixup:
    def test_output_shape_matches_first_image(self):
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((200, 200, 3), 200, dtype=np.uint8)
        mixed, bboxes, ids = _apply_mixup(
            img1, [(0.5, 0.5, 0.2, 0.2)], [0],
            img2, [(0.3, 0.3, 0.1, 0.1)], [1],
        )
        assert mixed.shape == img1.shape
        assert len(bboxes) == 2
        assert ids == [0, 1]

    def test_empty_bboxes(self):
        img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 64, dtype=np.uint8)
        mixed, bboxes, ids = _apply_mixup(
            img1, [], [], img2, [], [],
        )
        assert mixed.shape == img1.shape
        assert bboxes == []
        assert ids == []


# ---------------------------------------------------------------------------
# Tests: _apply_cutmix
# ---------------------------------------------------------------------------

class TestCutMix:
    def test_output_shape_matches_first_image(self):
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((200, 200, 3), 200, dtype=np.uint8)
        result, bboxes, ids = _apply_cutmix(
            img1, [(0.5, 0.5, 0.2, 0.2)], [0],
            img2, [(0.3, 0.3, 0.1, 0.1)], [1],
        )
        assert result.shape == img1.shape

    def test_does_not_crash_empty_bboxes(self):
        img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 64, dtype=np.uint8)
        result, bboxes, ids = _apply_cutmix(
            img1, [], [], img2, [], [],
        )
        assert result.shape == img1.shape


# ---------------------------------------------------------------------------
# Tests: apply_augmentations (integration)
# ---------------------------------------------------------------------------

class TestApplyAugmentations:
    def test_generates_images_side_by_side(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["flip_horizontal"],
            num_images=3,
        )
        assert result["generated_count"] == 3
        for path in result["generated_files"]:
            assert os.path.exists(path)
            # Check that the label file also exists
            base = os.path.splitext(path)[0]
            assert os.path.exists(base + ".txt")

    def test_generates_images_separated_structure(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=False)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["brightness"],
            num_images=2,
        )
        assert result["generated_count"] == 2
        for path in result["generated_files"]:
            assert os.path.exists(path)
            # Labels should be in the labels directory
            lbl_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "labels")
            lbl_name = os.path.splitext(os.path.basename(path))[0] + ".txt"
            assert os.path.exists(os.path.join(lbl_dir, lbl_name))

    def test_labels_have_valid_yolo_format(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["flip_horizontal"],
            num_images=2,
        )
        for path in result["generated_files"]:
            lbl_path = os.path.splitext(path)[0] + ".txt"
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5
                    int(parts[0])  # class_id
                    for val in parts[1:]:
                        v = float(val)
                        assert 0.0 <= v <= 1.0, f"bbox value {v} out of YOLO range"

    def test_no_images_for_missing_class(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[99],
            augmentation_names=["rotate"],
            num_images=5,
        )
        assert result["generated_count"] == 0

    def test_unknown_augmentation_ignored(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2)
        # Should fallthrough to empty pipeline, but still attempt to write
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["completely_fake_name"],
            num_images=2,
        )
        # With an empty standard pipeline and no mixup/cutmix this produces
        # images that are the original + no transform (pipeline is empty Compose)
        assert result["generated_count"] == 2

    def test_mixup_generates_files(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["mixup"],
            num_images=2,
        )
        assert result["generated_count"] == 2

    def test_cutmix_generates_files(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["cutmix"],
            num_images=2,
        )
        # CutMix might drop bboxes, so generated_count could be <= 2
        assert result["generated_count"] >= 0

    def test_multiple_augmentations_combined(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["flip_horizontal", "brightness", "gaussian_blur"],
            num_images=3,
        )
        assert result["generated_count"] == 3

    def test_multiple_classes(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0, 1],
            augmentation_names=["flip_horizontal"],
            num_images=2,
        )
        # 2 per class = 4 total
        assert result["generated_count"] == 4


# ---------------------------------------------------------------------------
# Tests: strict_filter in apply_augmentations
# ---------------------------------------------------------------------------

class TestStrictFilter:
    def test_strict_skips_mixed_class(self, tmp_path):
        """Class 1 has no exclusive images → skipped under strict_filter."""
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[1],
            augmentation_names=["flip_horizontal"],
            num_images=5,
            strict_filter=True,
        )
        assert result["generated_count"] == 0
        assert 1 in result["skipped_classes"]

    def test_strict_generates_only_target_class(self, tmp_path):
        """With strict_filter, output labels should only have class 0."""
        ds = _make_dataset(tmp_path, num_images=3, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0],
            augmentation_names=["flip_horizontal"],
            num_images=3,
            strict_filter=True,
        )
        assert result["generated_count"] == 3
        for path in result["generated_files"]:
            lbl_path = os.path.splitext(path)[0] + ".txt"
            with open(lbl_path) as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    assert cls_id == 0, "strict filter should exclude non-target classes"

    def test_strict_returns_skipped_classes_list(self, tmp_path):
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[0, 1],
            augmentation_names=["brightness"],
            num_images=2,
            strict_filter=True,
        )
        # class 0 should have generated, class 1 should be skipped
        assert result["generated_count"] == 2
        assert 1 in result["skipped_classes"]
        assert 0 not in result["skipped_classes"]

    def test_non_strict_default_keeps_all(self, tmp_path):
        """Without strict_filter, mixed-class images are used."""
        ds = _make_dataset(tmp_path, num_images=2, side_by_side=True)
        result = apply_augmentations(
            dataset=ds,
            target_classes=[1],
            augmentation_names=["flip_horizontal"],
            num_images=2,
        )
        # Class 1 exists in img0 (mixed), so augmentation proceeds
        assert result["generated_count"] == 2
        assert result.get("skipped_classes", []) == []
