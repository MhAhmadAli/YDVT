"""
Augmenter module for YDVT.

Provides data augmentation capabilities for YOLO-format datasets.
Supports per-class augmentations to balance imbalanced datasets.

Augmentations are powered by the `albumentations` library with
bounding-box-aware transforms so that annotations stay consistent.
"""

import os
import random
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations import BboxParams

from ydvt.parser import Dataset, ImageRecord, BBox


# ---------------------------------------------------------------------------
# Augmentation catalogue
# ---------------------------------------------------------------------------

# Each entry maps a user-facing name to a factory that returns an
# albumentations transform given optional keyword parameters.

AUGMENTATION_CATALOGUE: Dict[str, callable] = {
    "rotate": lambda **kw: A.Rotate(
        limit=kw.get("limit", 30), p=1.0, border_mode=cv2.BORDER_CONSTANT
    ),
    "flip_horizontal": lambda **kw: A.HorizontalFlip(p=1.0),
    "random_crop": lambda **kw: A.RandomResizedCrop(
        size=(kw.get("height", 640), kw.get("width", 640)),
        scale=(kw.get("scale_min", 0.5), kw.get("scale_max", 1.0)),
        p=1.0,
    ),
    "resize": lambda **kw: A.RandomScale(
        scale_limit=(kw.get("scale_min", -0.3), kw.get("scale_max", 0.3)),
        p=1.0,
    ),
    "translate": lambda **kw: A.Affine(
        translate_percent={
            "x": (-kw.get("tx", 0.1), kw.get("tx", 0.1)),
            "y": (-kw.get("ty", 0.1), kw.get("ty", 0.1)),
        },
        p=1.0,
    ),
    "brightness": lambda **kw: A.RandomBrightnessContrast(
        brightness_limit=(kw.get("brightness_min", -0.2), kw.get("brightness_max", 0.2)),
        contrast_limit=0,
        p=1.0,
    ),
    "contrast": lambda **kw: A.RandomBrightnessContrast(
        brightness_limit=0,
        contrast_limit=(kw.get("contrast_min", -0.2), kw.get("contrast_max", 0.2)),
        p=1.0,
    ),
    "saturation": lambda **kw: A.HueSaturationValue(
        hue_shift_limit=0,
        sat_shift_limit=kw.get("sat_shift", 30),
        val_shift_limit=0,
        p=1.0,
    ),
    "hue": lambda **kw: A.HueSaturationValue(
        hue_shift_limit=kw.get("hue_shift", 20),
        sat_shift_limit=0,
        val_shift_limit=0,
        p=1.0,
    ),
    "gaussian_blur": lambda **kw: A.GaussianBlur(
        blur_limit=(kw.get("blur_min", 3), kw.get("blur_max", 7)),
        p=1.0,
    ),
    "gaussian_noise": lambda **kw: A.GaussNoise(
        std_range=(kw.get("noise_min", 0.02), kw.get("noise_max", 0.1)),
        p=1.0,
    ),
    "cutout": lambda **kw: A.CoarseDropout(
        num_holes_range=(kw.get("num_holes_min", 1), kw.get("num_holes_max", 8)),
        hole_height_range=(kw.get("hole_h_min", 20), kw.get("hole_h_max", 60)),
        hole_width_range=(kw.get("hole_w_min", 20), kw.get("hole_w_max", 60)),
        fill=0,
        p=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_pipeline(augmentation_names: List[str],
                    params: Optional[Dict] = None) -> A.Compose:
    """Build an albumentations Compose pipeline from a list of names."""
    params = params or {}
    transforms = []
    for name in augmentation_names:
        factory = AUGMENTATION_CATALOGUE.get(name)
        if factory is not None:
            kw = params.get(name, {})
            transforms.append(factory(**kw))
    return A.Compose(
        transforms,
        bbox_params=BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.1,
        ),
    )


def _images_for_class(dataset: Dataset, class_id: int,
                       strict: bool = False) -> List[ImageRecord]:
    """
    Return images containing the given class.

    Args:
        dataset: Parsed Dataset object.
        class_id: Target class ID.
        strict: If True, only return images where EVERY bounding box
                belongs to class_id (no other classes present).
                If False (default), any image containing at least one
                bbox of class_id is returned.
    """
    if strict:
        return [
            img for img in dataset.images
            if img.bboxes and all(b.class_id == class_id for b in img.bboxes)
        ]
    return [img for img in dataset.images if any(b.class_id == class_id for b in img.bboxes)]


def _resolve_output_dirs(image_path: str):
    """
    Determine where to save the augmented image and label.
    Respects both side-by-side and images/labels directory structures.
    Returns (images_dir, labels_dir).
    """
    parent = os.path.dirname(image_path)
    if os.path.basename(parent) == "images":
        images_dir = parent
        labels_dir = os.path.join(os.path.dirname(parent), "labels")
    else:
        images_dir = parent
        labels_dir = parent
    return images_dir, labels_dir


def _write_yolo_label(label_path: str, bboxes: list, class_ids: list):
    """Write bounding boxes in YOLO format to a text file."""
    with open(label_path, "w", encoding="utf-8") as f:
        for bbox, cls_id in zip(bboxes, class_ids):
            x_c, y_c, w, h = bbox
            f.write(f"{int(cls_id)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Mixup / CutMix  (custom, not in the standard pipeline)
# ---------------------------------------------------------------------------

def _apply_mixup(img1: np.ndarray, bboxes1: list, ids1: list,
                 img2: np.ndarray, bboxes2: list, ids2: list,
                 alpha: float = 0.5) -> Tuple[np.ndarray, list, list]:
    """Blend two images and merge their annotations."""
    # Resize img2 to match img1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.7)
    mixed = (lam * img1.astype(np.float32) + (1 - lam) * img2_resized.astype(np.float32)).astype(np.uint8)
    merged_bboxes = list(bboxes1) + list(bboxes2)
    merged_ids = list(ids1) + list(ids2)
    return mixed, merged_bboxes, merged_ids


def _apply_cutmix(img1: np.ndarray, bboxes1: list, ids1: list,
                  img2: np.ndarray, bboxes2: list, ids2: list) -> Tuple[np.ndarray, list, list]:
    """Paste a random crop of img2 onto img1 and merge annotations."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))

    # Random rectangle
    cut_w = int(w * np.random.uniform(0.2, 0.5))
    cut_h = int(h * np.random.uniform(0.2, 0.5))
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)

    result = img1.copy()
    result[y1:y2, x1:x2] = img2_resized[y1:y2, x1:x2]

    # Keep bboxes from img1 that are NOT fully inside the cut region, and
    # keep bboxes from img2 that are fully inside the cut region.
    kept_bboxes = []
    kept_ids = []

    for bbox, cls_id in zip(bboxes1, ids1):
        bx, by, bw, bh = bbox
        # Convert to pixel coords
        bx1 = (bx - bw / 2) * w
        by1 = (by - bh / 2) * h
        bx2 = (bx + bw / 2) * w
        by2 = (by + bh / 2) * h
        # Keep if not fully inside the cut rectangle
        if not (bx1 >= x1 and by1 >= y1 and bx2 <= x2 and by2 <= y2):
            kept_bboxes.append(bbox)
            kept_ids.append(cls_id)

    for bbox, cls_id in zip(bboxes2, ids2):
        bx, by, bw, bh = bbox
        bx1 = (bx - bw / 2) * w
        by1 = (by - bh / 2) * h
        bx2 = (bx + bw / 2) * w
        by2 = (by + bh / 2) * h
        # Keep if at least partially inside the cut rectangle
        if bx2 > x1 and bx1 < x2 and by2 > y1 and by1 < y2:
            # Clip to cut region, convert back to YOLO
            cbx1 = max(bx1, x1)
            cby1 = max(by1, y1)
            cbx2 = min(bx2, x2)
            cby2 = min(by2, y2)
            nw = (cbx2 - cbx1) / w
            nh = (cby2 - cby1) / h
            nx = (cbx1 + cbx2) / 2 / w
            ny = (cby1 + cby2) / 2 / h
            if nw > 0.01 and nh > 0.01:
                kept_bboxes.append((nx, ny, nw, nh))
                kept_ids.append(cls_id)

    return result, kept_bboxes, kept_ids


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def apply_augmentations(
    dataset: Dataset,
    target_classes: List[int],
    augmentation_names: List[str],
    num_images: int = 5,
    params: Optional[Dict] = None,
    strict_filter: bool = False,
) -> Dict:
    """
    Apply selected augmentations to images of the target classes.

    For each target class, randomly sample source images that contain
    that class and generate *num_images* augmented copies.

    For Mixup and CutMix, a secondary image is sampled randomly from
    the whole dataset (or the strictly filtered pool when strict_filter
    is True).

    Args:
        dataset: Parsed Dataset object.
        target_classes: List of class IDs to augment.
        augmentation_names: List of augmentation names to apply.
        num_images: Number of augmented images to generate per class.
        params: Optional per-augmentation parameter overrides.
        strict_filter: If True, only select source images where EVERY
                       bounding box belongs to the target class. This
                       prevents non-target class counts from growing.
                       Defaults to False.

    Returns:
        dict with ``generated_count`` (int), ``generated_files``
        (list of created file paths), and ``skipped_classes`` (list of
        class IDs that had no eligible images under strict filtering).
    """
    params = params or {}

    # Separate standard augmentations from special ones
    standard_names = [n for n in augmentation_names if n in AUGMENTATION_CATALOGUE]
    do_mixup = "mixup" in augmentation_names
    do_cutmix = "cutmix" in augmentation_names

    pipeline = _build_pipeline(standard_names, params) if standard_names else None

    generated_files: List[str] = []
    skipped_classes: List[int] = []

    for class_id in target_classes:
        source_images = _images_for_class(dataset, class_id, strict=strict_filter)
        if not source_images:
            skipped_classes.append(class_id)
            continue

        for i in range(num_images):
            src = random.choice(source_images)
            img = cv2.imread(src.image_path)
            if img is None:
                continue

            bboxes = [(b.x_center, b.y_center, b.width, b.height) for b in src.bboxes]
            class_ids = [b.class_id for b in src.bboxes]

            # Apply standard pipeline
            if pipeline is not None and bboxes:
                try:
                    result = pipeline(image=img, bboxes=bboxes, class_ids=class_ids)
                    img = result["image"]
                    bboxes = result["bboxes"]
                    class_ids = result["class_ids"]
                except Exception:
                    # If the transform fails (e.g. all bboxes clipped out), skip
                    continue

            # Pool for secondary images in Mixup / CutMix
            secondary_pool = (
                _images_for_class(dataset, class_id, strict=True)
                if strict_filter else dataset.images
            )

            # Apply Mixup
            if do_mixup and len(secondary_pool) > 1:
                other = random.choice(secondary_pool)
                other_img = cv2.imread(other.image_path)
                if other_img is not None:
                    other_bboxes = [(b.x_center, b.y_center, b.width, b.height) for b in other.bboxes]
                    other_ids = [b.class_id for b in other.bboxes]
                    img, bboxes, class_ids = _apply_mixup(
                        img, bboxes, class_ids,
                        other_img, other_bboxes, other_ids,
                    )

            # Apply CutMix
            if do_cutmix and len(secondary_pool) > 1:
                other = random.choice(secondary_pool)
                other_img = cv2.imread(other.image_path)
                if other_img is not None:
                    other_bboxes = [(b.x_center, b.y_center, b.width, b.height) for b in other.bboxes]
                    other_ids = [b.class_id for b in other.bboxes]
                    img, bboxes, class_ids = _apply_cutmix(
                        img, bboxes, class_ids,
                        other_img, other_bboxes, other_ids,
                    )

            if not bboxes:
                continue

            # Write output
            uid = uuid.uuid4().hex[:8]
            images_dir, labels_dir = _resolve_output_dirs(src.image_path)
            os.makedirs(labels_dir, exist_ok=True)

            ext = os.path.splitext(src.image_path)[1]
            out_name = f"aug_{class_id}_{uid}"
            out_img_path = os.path.join(images_dir, out_name + ext)
            out_lbl_path = os.path.join(labels_dir, out_name + ".txt")

            cv2.imwrite(out_img_path, img)
            _write_yolo_label(out_lbl_path, bboxes, class_ids)
            generated_files.append(out_img_path)

    return {
        "generated_count": len(generated_files),
        "generated_files": generated_files,
        "skipped_classes": skipped_classes,
    }


def list_available_augmentations() -> List[Dict]:
    """Return metadata about all available augmentations for the frontend."""
    descriptions = {
        "rotate": "Rotate the image by a random angle",
        "flip_horizontal": "Flip the image horizontally",
        "random_crop": "Randomly crop and resize the image",
        "resize": "Randomly scale the image up or down",
        "translate": "Shift the image horizontally and vertically",
        "brightness": "Randomly adjust brightness",
        "contrast": "Randomly adjust contrast",
        "saturation": "Randomly adjust colour saturation",
        "hue": "Randomly shift the hue channel",
        "gaussian_blur": "Apply Gaussian blur",
        "gaussian_noise": "Add random Gaussian noise",
        "cutout": "Randomly erase rectangular regions (Random Erasing)",
        "mixup": "Blend two images together (alpha blending)",
        "cutmix": "Paste a crop of another image onto this one",
    }
    result = []
    for name in list(AUGMENTATION_CATALOGUE.keys()) + ["mixup", "cutmix"]:
        result.append({
            "name": name,
            "label": name.replace("_", " ").title(),
            "description": descriptions.get(name, ""),
        })
    return result
