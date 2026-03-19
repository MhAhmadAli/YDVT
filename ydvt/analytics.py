"""
Analytics module for computing dataset statistics.

Provides the core ``compute_analytics`` function that always computes
class-distribution and split-distribution (the defaults) and optionally
computes 13 additional advanced metrics when their flags are set.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import hashlib
import math
import os
import re

import numpy as np

from .parser import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known split directory names (case-insensitive matching)
_SPLIT_NAMES = {"train", "valid", "validation", "val", "test"}

# Bbox area thresholds (relative to image, i.e. w*h in 0..1)
_SMALL_AREA = 0.01   # < 1% of image
_LARGE_AREA = 0.10   # > 10% of image

# Heatmap grid resolution
_HEATMAP_GRID = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_split(image_path: str) -> str:
    """
    Detect the dataset split from the image file path.

    Looks for known directory names (train, valid, val, validation, test) in
    the path components. Returns the normalised split name, or 'unassigned'
    if no split directory is found.
    """
    parts = image_path.replace("\\", "/").lower().split("/")
    for part in parts:
        if part in _SPLIT_NAMES:
            # Normalise validation/val → valid
            if part in ("validation", "val"):
                return "valid"
            return part
    return "unassigned"


def _median(values: List[float]) -> float:
    """Return the median of a list of numbers."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _std_dev(values: List[float], mean: float) -> float:
    """Return the population standard deviation."""
    if not values:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _file_md5(path: str) -> Optional[str]:
    """Return the MD5 hex-digest of a file, or None on error."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _kmeans_1d(data: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Minimal 2-D K-Means for (w, h) bbox clusters. Returns k centroids."""
    if len(data) == 0:
        return np.zeros((k, 2))
    if len(data) <= k:
        padded = np.zeros((k, 2))
        padded[: len(data)] = data
        return padded

    # Deterministic init: evenly spaced indices
    indices = np.linspace(0, len(data) - 1, k, dtype=int)
    centroids = data[indices].copy()

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        dists = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = data[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids


# ---------------------------------------------------------------------------
# All option keys recognised by compute_analytics
# ---------------------------------------------------------------------------
ALL_OPTION_KEYS = [
    "images_per_class",
    "bbox_count_per_image",
    "bbox_size_dist",
    "bbox_aspect_ratio",
    "location_heatmaps",
    "image_resolution_dist",
    "label_density",
    "co_occurrence_matrix",
    "annotation_completeness",
    "duplicate_detection",
    "label_imbalance",
    "outlier_detection",
    "anchor_analysis",
]


# ---------------------------------------------------------------------------
# Main analytics function
# ---------------------------------------------------------------------------

def compute_analytics(
    dataset: Dataset,
    options: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Computes analytics from the parsed dataset.

    Parameters
    ----------
    dataset : Dataset
        Parsed YOLO dataset.
    options : dict, optional
        Mapping of analytics option names to booleans.  Only the options
        set to ``True`` are computed.  ``None`` means *defaults only*
        (class distribution + split distribution).

    Returns
    -------
    dict
        Structured analytics results keyed by metric name.
    """
    if options is None:
        options = {}

    total_images = len(dataset.images)
    total_bboxes = 0
    images_with_annotations = 0

    # Per class metrics
    class_counts: Dict[int, int] = {cls_id: 0 for cls_id in dataset.classes}
    class_sizes: Dict[int, Dict[str, float]] = {
        cls_id: {"w_sum": 0.0, "h_sum": 0.0, "count": 0}
        for cls_id in dataset.classes
    }

    # Track unmapped classes
    unmapped_classes: Set[int] = set()

    # Split tracking
    split_counts: Dict[str, Dict[str, int]] = {}

    # Optional accumulators — initialised only when needed
    want_ipc = options.get("images_per_class", False)
    want_bpi = options.get("bbox_count_per_image", False)
    want_bsd = options.get("bbox_size_dist", False)
    want_bar = options.get("bbox_aspect_ratio", False)
    want_hm  = options.get("location_heatmaps", False)
    want_ird = options.get("image_resolution_dist", False)
    want_ld  = options.get("label_density", False)
    want_com = options.get("co_occurrence_matrix", False)
    want_ac  = options.get("annotation_completeness", False)
    want_dd  = options.get("duplicate_detection", False)
    want_li  = options.get("label_imbalance", False)
    want_od  = options.get("outlier_detection", False)
    want_aa  = options.get("anchor_analysis", False)

    # images_per_class: track unique images per class
    images_per_class: Dict[int, int] = (
        {cls_id: 0 for cls_id in dataset.classes} if want_ipc else {}
    )

    # bbox_count_per_image
    bbox_counts_per_img: List[int] = [] if want_bpi else []

    # bbox_size_dist
    bbox_areas: List[float] = []

    # bbox_aspect_ratio
    bbox_ratios: List[float] = []

    # location_heatmaps
    heatmap = np.zeros((_HEATMAP_GRID, _HEATMAP_GRID), dtype=int) if want_hm else None

    # image_resolution_dist
    resolution_counts: Dict[str, int] = {} if want_ird else {}

    # co_occurrence_matrix
    all_class_ids_for_com: List[Set[int]] = []

    # All bbox w/h for anchor analysis & outlier detection
    all_bbox_wh: List[Tuple[float, float]] = []

    # duplicate_detection
    hash_to_paths: Dict[str, List[str]] = {} if want_dd else {}

    # -----------------------------------------------------------------------
    # Main loop over images
    # -----------------------------------------------------------------------
    for record in dataset.images:
        split = _detect_split(record.image_path)
        if split not in split_counts:
            split_counts[split] = {"images": 0, "bboxes": 0}
        split_counts[split]["images"] += 1

        if want_ird and record.width > 0 and record.height > 0:
            res_key = f"{record.width}x{record.height}"
            resolution_counts[res_key] = resolution_counts.get(res_key, 0) + 1

        if want_dd:
            md5 = _file_md5(record.image_path)
            if md5:
                hash_to_paths.setdefault(md5, []).append(record.image_path)

        num_bboxes = len(record.bboxes)
        if want_bpi:
            bbox_counts_per_img.append(num_bboxes)

        classes_in_image: Set[int] = set()

        if record.bboxes:
            images_with_annotations += 1
            total_bboxes += num_bboxes
            split_counts[split]["bboxes"] += num_bboxes

            for bbox in record.bboxes:
                cid = bbox.class_id
                classes_in_image.add(cid)

                if cid in dataset.classes:
                    class_counts[cid] += 1
                    class_sizes[cid]["w_sum"] += bbox.width
                    class_sizes[cid]["h_sum"] += bbox.height
                    class_sizes[cid]["count"] += 1
                else:
                    unmapped_classes.add(cid)
                    if cid not in class_counts:
                        class_counts[cid] = 1
                        class_sizes[cid] = {
                            "w_sum": bbox.width,
                            "h_sum": bbox.height,
                            "count": 1,
                        }
                    else:
                        class_counts[cid] += 1
                        class_sizes[cid]["w_sum"] += bbox.width
                        class_sizes[cid]["h_sum"] += bbox.height
                        class_sizes[cid]["count"] += 1

                # Bbox area
                area = bbox.width * bbox.height
                if want_bsd:
                    bbox_areas.append(area)

                # Aspect ratio
                if want_bar and bbox.height > 0:
                    bbox_ratios.append(bbox.width / bbox.height)

                # Heatmap
                if want_hm:
                    col = min(int(bbox.x_center * _HEATMAP_GRID), _HEATMAP_GRID - 1)
                    row = min(int(bbox.y_center * _HEATMAP_GRID), _HEATMAP_GRID - 1)
                    heatmap[row, col] += 1

                # Collect for anchor / outlier
                if want_aa or want_od:
                    all_bbox_wh.append((bbox.width, bbox.height))

        if want_ipc:
            for cid in classes_in_image:
                if cid in images_per_class:
                    images_per_class[cid] += 1

        if want_com:
            all_class_ids_for_com.append(classes_in_image)

    # -----------------------------------------------------------------------
    # Finalize default metrics
    # -----------------------------------------------------------------------
    avg_sizes = {}
    for cls_id, stats in class_sizes.items():
        if stats["count"] > 0:
            avg_sizes[cls_id] = {
                "w": stats["w_sum"] / stats["count"],
                "h": stats["h_sum"] / stats["count"],
            }
        else:
            avg_sizes[cls_id] = {"w": 0.0, "h": 0.0}

    split_distribution = {}
    for split_name, counts in sorted(split_counts.items()):
        pct = (counts["images"] / total_images * 100) if total_images > 0 else 0.0
        split_distribution[split_name] = {
            "images": counts["images"],
            "bboxes": counts["bboxes"],
            "percentage": round(pct, 1),
        }

    result: Dict[str, Any] = {
        "summary": {
            "total_images": total_images,
            "images_with_annotations": images_with_annotations,
            "total_bboxes": total_bboxes,
            "unmapped_classes": list(unmapped_classes),
        },
        "class_distribution": {
            dataset.classes.get(cls_id, f"Class {cls_id}"): count
            for cls_id, count in class_counts.items()
        },
        "avg_bbox_sizes": {
            dataset.classes.get(cls_id, f"Class {cls_id}"): sizes
            for cls_id, sizes in avg_sizes.items()
        },
        "split_distribution": split_distribution,
    }

    # -----------------------------------------------------------------------
    # Optional analytics
    # -----------------------------------------------------------------------

    if want_ipc:
        result["images_per_class"] = {
            dataset.classes.get(cid, f"Class {cid}"): cnt
            for cid, cnt in images_per_class.items()
        }

    if want_bpi and bbox_counts_per_img:
        mean_bpi = sum(bbox_counts_per_img) / len(bbox_counts_per_img)
        result["bbox_count_per_image"] = {
            "min": min(bbox_counts_per_img),
            "max": max(bbox_counts_per_img),
            "mean": round(mean_bpi, 2),
            "median": _median([float(v) for v in bbox_counts_per_img]),
        }

    if want_bsd and bbox_areas:
        small = sum(1 for a in bbox_areas if a < _SMALL_AREA)
        medium = sum(1 for a in bbox_areas if _SMALL_AREA <= a <= _LARGE_AREA)
        large = sum(1 for a in bbox_areas if a > _LARGE_AREA)
        result["bbox_size_dist"] = {
            "small": small,
            "medium": medium,
            "large": large,
            "thresholds": {"small_lt": _SMALL_AREA, "large_gt": _LARGE_AREA},
        }

    if want_bar and bbox_ratios:
        mean_r = sum(bbox_ratios) / len(bbox_ratios)
        result["bbox_aspect_ratio"] = {
            "min": round(min(bbox_ratios), 3),
            "max": round(max(bbox_ratios), 3),
            "mean": round(mean_r, 3),
            "median": round(_median(bbox_ratios), 3),
        }

    if want_hm and heatmap is not None:
        result["location_heatmaps"] = heatmap.tolist()

    if want_ird and resolution_counts:
        result["image_resolution_dist"] = dict(
            sorted(resolution_counts.items(), key=lambda x: x[1], reverse=True)
        )

    if want_ld:
        avg_bboxes = (total_bboxes / total_images) if total_images > 0 else 0.0
        result["label_density"] = {
            "avg_bboxes_per_image": round(avg_bboxes, 2),
            "total_bboxes": total_bboxes,
            "total_images": total_images,
        }

    if want_com:
        # Build sorted list of all class IDs that appeared
        all_cids = sorted(
            {cid for img_cids in all_class_ids_for_com for cid in img_cids}
        )
        matrix: Dict[str, Dict[str, int]] = {}
        for cid_a in all_cids:
            name_a = dataset.classes.get(cid_a, f"Class {cid_a}")
            matrix[name_a] = {}
            for cid_b in all_cids:
                name_b = dataset.classes.get(cid_b, f"Class {cid_b}")
                count = sum(
                    1
                    for img_cids in all_class_ids_for_com
                    if cid_a in img_cids and cid_b in img_cids
                )
                matrix[name_a][name_b] = count
        result["co_occurrence_matrix"] = matrix

    if want_ac:
        images_without = total_images - images_with_annotations
        pct_annotated = (
            (images_with_annotations / total_images * 100) if total_images > 0 else 0.0
        )
        result["annotation_completeness"] = {
            "images_with_annotations": images_with_annotations,
            "images_without_annotations": images_without,
            "percentage_annotated": round(pct_annotated, 1),
        }

    if want_dd:
        duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
        result["duplicate_detection"] = {
            "duplicate_groups": len(duplicates),
            "total_duplicate_images": sum(len(p) for p in duplicates.values()),
            "groups": {h: paths for h, paths in duplicates.items()},
        }

    if want_li and class_counts:
        counts = list(class_counts.values())
        max_c = max(counts) if counts else 0
        min_c = min(counts) if counts else 0
        ratio = (max_c / min_c) if min_c > 0 else float("inf")
        mean_c = sum(counts) / len(counts) if counts else 0
        std_c = _std_dev([float(c) for c in counts], mean_c)
        result["label_imbalance"] = {
            "max_count": max_c,
            "min_count": min_c,
            "imbalance_ratio": round(ratio, 2),
            "mean": round(mean_c, 2),
            "std_dev": round(std_c, 2),
        }

    if want_od and all_bbox_wh:
        widths = [w for w, _ in all_bbox_wh]
        heights = [h for _, h in all_bbox_wh]
        areas = [w * h for w, h in all_bbox_wh]
        ratios = [w / h for w, h in all_bbox_wh if h > 0]

        mean_area = sum(areas) / len(areas)
        std_area = _std_dev(areas, mean_area)
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0
        std_ratio = _std_dev(ratios, mean_ratio) if ratios else 0

        outlier_bboxes = []
        for i, (w, h) in enumerate(all_bbox_wh):
            a = w * h
            r = w / h if h > 0 else 0
            reasons = []
            if abs(a - mean_area) > 2 * std_area and std_area > 0:
                reasons.append("extreme_area")
            if ratios and abs(r - mean_ratio) > 2 * std_ratio and std_ratio > 0:
                reasons.append("extreme_ratio")
            if reasons:
                outlier_bboxes.append({
                    "index": i,
                    "width": round(w, 4),
                    "height": round(h, 4),
                    "area": round(a, 6),
                    "ratio": round(r, 3),
                    "reasons": reasons,
                })

        result["outlier_detection"] = {
            "total_outliers": len(outlier_bboxes),
            "area_mean": round(mean_area, 6),
            "area_std": round(std_area, 6),
            "ratio_mean": round(mean_ratio, 3),
            "ratio_std": round(std_ratio, 3),
            "outliers": outlier_bboxes[:50],  # cap output
        }

    if want_aa and all_bbox_wh:
        arr = np.array(all_bbox_wh, dtype=float)
        k = min(5, len(arr))  # up to 5 anchor clusters
        centroids = _kmeans_1d(arr, k)
        # Sort by area (w*h)
        areas = centroids[:, 0] * centroids[:, 1]
        order = np.argsort(areas)
        centroids = centroids[order]
        result["anchor_analysis"] = {
            "k": k,
            "anchors": [
                {"width": round(float(c[0]), 4), "height": round(float(c[1]), 4)}
                for c in centroids
                if c[0] > 0 or c[1] > 0
            ],
        }

    return result
