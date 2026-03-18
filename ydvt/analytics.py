from typing import Dict, Any, List
from .parser import Dataset

def compute_analytics(dataset: Dataset) -> Dict[str, Any]:
    """
    Computes analytics from the parsed dataset.
    Returns a dictionary structured for JSON serialization/GUI rendering
    and terminal visualization.
    """
    total_images = len(dataset.images)
    total_bboxes = 0
    images_with_annotations = 0
    
    # Per class metrics
    class_counts = {cls_id: 0 for cls_id in dataset.classes.keys()}
    
    # Bbox size metrics (using absolute sizes for GUI plotting)
    # We will compute average width/height relative to image sizes, or absolute if image dimensions exist.
    class_sizes = {cls_id: {"w_sum": 0.0, "h_sum": 0.0, "count": 0} for cls_id in dataset.classes.keys()}
    
    # Track unmapped classes just in case
    unmapped_classes = set()

    for record in dataset.images:
        if record.bboxes:
            images_with_annotations += 1
            total_bboxes += len(record.bboxes)
            
            for bbox in record.bboxes:
                if bbox.class_id in dataset.classes:
                    class_counts[bbox.class_id] += 1
                    class_sizes[bbox.class_id]["w_sum"] += bbox.width # YOLO format stores relative sizes [0,1]
                    class_sizes[bbox.class_id]["h_sum"] += bbox.height
                    class_sizes[bbox.class_id]["count"] += 1
                else:
                    unmapped_classes.add(bbox.class_id)
                    # Initialize dynamic class if not in classes.txt
                    if bbox.class_id not in class_counts:
                        class_counts[bbox.class_id] = 1
                        class_sizes[bbox.class_id] = {"w_sum": bbox.width, "h_sum": bbox.height, "count": 1}
                    else:
                        class_counts[bbox.class_id] += 1
                        class_sizes[bbox.class_id]["w_sum"] += bbox.width
                        class_sizes[bbox.class_id]["h_sum"] += bbox.height
                        class_sizes[bbox.class_id]["count"] += 1

    # Finalize average sizes
    avg_sizes = {}
    for cls_id, stats in class_sizes.items():
        if stats["count"] > 0:
            avg_sizes[cls_id] = {
                "w": stats["w_sum"] / stats["count"],
                "h": stats["h_sum"] / stats["count"]
            }
        else:
            avg_sizes[cls_id] = {"w": 0.0, "h": 0.0}

    # Format the output metrics
    return {
        "summary": {
            "total_images": total_images,
            "images_with_annotations": images_with_annotations,
            "total_bboxes": total_bboxes,
            "unmapped_classes": list(unmapped_classes)
        },
        "class_distribution": {
            dataset.classes.get(cls_id, f"Class {cls_id}"): count
            for cls_id, count in class_counts.items()
        },
        "avg_bbox_sizes": {
            dataset.classes.get(cls_id, f"Class {cls_id}"): sizes
            for cls_id, sizes in avg_sizes.items()
        }
    }
