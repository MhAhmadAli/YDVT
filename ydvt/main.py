"""
CLI entrypoint for the YDVT tool.

Routes to the Web GUI (``--gui``), the augmentation wizard (``--augment``),
or the default terminal analytics output.  Optional analytics flags enable
additional metrics beyond the default class & split distributions.
"""

import argparse
import os
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ydvt.parser import parse_yolo_dataset
from ydvt.analytics import compute_analytics, ALL_OPTION_KEYS

# Maps CLI flag dest names → analytics option keys
_FLAG_TO_OPTION = {
    "images_per_class":    "images_per_class",
    "bbox_count_per_image":"bbox_count_per_image",
    "bbox_size_dist":      "bbox_size_dist",
    "bbox_aspect_ratio":   "bbox_aspect_ratio",
    "location_heatmaps":   "location_heatmaps",
    "image_resolution_dist":"image_resolution_dist",
    "label_density":       "label_density",
    "co_occurrence_matrix":"co_occurrence_matrix",
    "annotation_completeness":"annotation_completeness",
    "duplicate_detection": "duplicate_detection",
    "label_imbalance":     "label_imbalance",
    "outlier_detection":   "outlier_detection",
    "anchor_analysis":     "anchor_analysis",
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_defaults(console: Console, analytics: dict):
    """Render the always-on analytics: summary, class dist, split dist."""
    summary = analytics["summary"]
    console.print()
    console.print(Panel(
        f"Total Images: {summary['total_images']}\n"
        f"Images w/ Annotations: {summary['images_with_annotations']}\n"
        f"Total BBoxes: {summary['total_bboxes']}\n"
        f"Unmapped Classes: {summary['unmapped_classes']}",
        title="[bold blue]Dataset Summary[/bold blue]",
        expand=False,
    ))

    # Class Distribution
    table = Table(title="Class Distribution")
    table.add_column("Class", justify="left", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Avg BBox Size (W × H)", justify="right", style="green")

    dist = analytics["class_distribution"]
    avg_sizes = analytics["avg_bbox_sizes"]

    for cls_name, count in dist.items():
        if isinstance(cls_name, int) and summary["unmapped_classes"]:
            name = f"Class {cls_name} (Missing from classes.txt)"
        else:
            name = str(cls_name)

        size = avg_sizes.get(cls_name)
        if not size and cls_name not in avg_sizes:
            for k in avg_sizes:
                if str(k) == str(cls_name) or (isinstance(cls_name, str) and cls_name.endswith(str(k))):
                    size = avg_sizes[k]
                    break

        size_str = f"{size['w']:.3f} x {size['h']:.3f}" if size else "0.000 x 0.000"
        table.add_row(name, str(count), size_str)

    console.print(table)

    # Split Distribution
    split_dist = analytics.get("split_distribution", {})
    if split_dist:
        st = Table(title="Split Distribution")
        st.add_column("Split", justify="left", style="cyan", no_wrap=True)
        st.add_column("Images", justify="right", style="magenta")
        st.add_column("BBoxes", justify="right", style="green")
        st.add_column("%", justify="right", style="yellow")
        for name, info in split_dist.items():
            st.add_row(
                name.capitalize(),
                str(info["images"]),
                str(info["bboxes"]),
                f"{info['percentage']}%",
            )
        console.print(st)


def _render_images_per_class(console: Console, data: dict):
    t = Table(title="Images per Class")
    t.add_column("Class", style="cyan")
    t.add_column("Images", justify="right", style="magenta")
    for cls, cnt in data.items():
        t.add_row(str(cls), str(cnt))
    console.print(t)


def _render_bbox_count_per_image(console: Console, data: dict):
    console.print(Panel(
        f"Min: {data['min']}   Max: {data['max']}   "
        f"Mean: {data['mean']}   Median: {data['median']}",
        title="[bold blue]BBox Count per Image[/bold blue]",
        expand=False,
    ))


def _render_bbox_size_dist(console: Console, data: dict):
    t = Table(title="BBox Size Distribution")
    t.add_column("Category", style="cyan")
    t.add_column("Count", justify="right", style="magenta")
    t.add_column("Threshold", justify="right", style="dim")
    thr = data["thresholds"]
    t.add_row("Small", str(data["small"]), f"area < {thr['small_lt']}")
    t.add_row("Medium", str(data["medium"]), f"{thr['small_lt']} ≤ area ≤ {thr['large_gt']}")
    t.add_row("Large", str(data["large"]), f"area > {thr['large_gt']}")
    console.print(t)


def _render_bbox_aspect_ratio(console: Console, data: dict):
    console.print(Panel(
        f"Min: {data['min']}   Max: {data['max']}   "
        f"Mean: {data['mean']}   Median: {data['median']}",
        title="[bold blue]BBox Aspect Ratio (W/H)[/bold blue]",
        expand=False,
    ))


def _render_location_heatmaps(console: Console, grid: list):
    t = Table(title="Object Location Heatmap (10×10 grid)")
    t.add_column("", style="dim", width=3)
    cols = len(grid[0]) if grid else 0
    for c in range(cols):
        t.add_column(str(c), justify="right", style="magenta", width=5)
    for r, row in enumerate(grid):
        max_val = max(max(row) for row in grid) if grid else 1
        cells = []
        for v in row:
            if v == 0:
                cells.append("[dim]·[/dim]")
            elif v >= max_val * 0.75:
                cells.append(f"[bold red]{v}[/bold red]")
            elif v >= max_val * 0.5:
                cells.append(f"[yellow]{v}[/yellow]")
            elif v >= max_val * 0.25:
                cells.append(f"[green]{v}[/green]")
            else:
                cells.append(str(v))
        t.add_row(str(r), *cells)
    console.print(t)


def _render_image_resolution_dist(console: Console, data: dict):
    t = Table(title="Image Resolution Distribution")
    t.add_column("Resolution (WxH)", style="cyan")
    t.add_column("Count", justify="right", style="magenta")
    for res, cnt in data.items():
        t.add_row(res, str(cnt))
    console.print(t)


def _render_label_density(console: Console, data: dict):
    console.print(Panel(
        f"Avg BBoxes/Image: {data['avg_bboxes_per_image']}   "
        f"Total BBoxes: {data['total_bboxes']}   "
        f"Total Images: {data['total_images']}",
        title="[bold blue]Label Density[/bold blue]",
        expand=False,
    ))


def _render_co_occurrence_matrix(console: Console, matrix: dict):
    classes = list(matrix.keys())
    t = Table(title="Class Co-occurrence Matrix")
    t.add_column("", style="cyan", no_wrap=True)
    for c in classes:
        t.add_column(c, justify="right", style="magenta", no_wrap=True)
    for cls_a in classes:
        row_vals = [str(matrix[cls_a].get(cls_b, 0)) for cls_b in classes]
        t.add_row(cls_a, *row_vals)
    console.print(t)


def _render_annotation_completeness(console: Console, data: dict):
    console.print(Panel(
        f"Annotated: {data['images_with_annotations']}   "
        f"Un-annotated: {data['images_without_annotations']}   "
        f"Coverage: {data['percentage_annotated']}%",
        title="[bold blue]Annotation Completeness[/bold blue]",
        expand=False,
    ))


def _render_duplicate_detection(console: Console, data: dict):
    console.print(Panel(
        f"Duplicate groups: {data['duplicate_groups']}   "
        f"Total duplicate images: {data['total_duplicate_images']}",
        title="[bold blue]Duplicate Image Detection[/bold blue]",
        expand=False,
    ))
    groups = data.get("groups", {})
    if groups:
        t = Table(title="Duplicate Groups (by MD5)")
        t.add_column("#", justify="right", style="dim")
        t.add_column("Files", style="cyan")
        for i, (h, paths) in enumerate(groups.items(), 1):
            basenames = ", ".join(os.path.basename(p) for p in paths)
            t.add_row(str(i), basenames)
            if i >= 20:
                t.add_row("…", f"({len(groups) - 20} more groups)")
                break
        console.print(t)


def _render_label_imbalance(console: Console, data: dict):
    ratio = data["imbalance_ratio"]
    ratio_str = f"{ratio}" if ratio != float("inf") else "∞"
    console.print(Panel(
        f"Max: {data['max_count']}   Min: {data['min_count']}   "
        f"Ratio (max/min): {ratio_str}\n"
        f"Mean: {data['mean']}   Std Dev: {data['std_dev']}",
        title="[bold blue]Label Imbalance[/bold blue]",
        expand=False,
    ))


def _render_outlier_detection(console: Console, data: dict):
    console.print(Panel(
        f"Outlier BBoxes: {data['total_outliers']}   "
        f"Area μ={data['area_mean']:.4f} σ={data['area_std']:.4f}   "
        f"Ratio μ={data['ratio_mean']:.3f} σ={data['ratio_std']:.3f}",
        title="[bold blue]Outlier Detection (>2σ)[/bold blue]",
        expand=False,
    ))
    outliers = data.get("outliers", [])
    if outliers:
        t = Table(title=f"Top Outliers (showing {len(outliers)})")
        t.add_column("#", justify="right", style="dim")
        t.add_column("W", justify="right", style="cyan")
        t.add_column("H", justify="right", style="cyan")
        t.add_column("Area", justify="right", style="magenta")
        t.add_column("Ratio", justify="right", style="green")
        t.add_column("Reason", style="yellow")
        for o in outliers[:20]:
            t.add_row(
                str(o["index"]),
                str(o["width"]),
                str(o["height"]),
                str(o["area"]),
                str(o["ratio"]),
                ", ".join(o["reasons"]),
            )
        console.print(t)


def _render_anchor_analysis(console: Console, data: dict):
    t = Table(title=f"Suggested Anchor Boxes (K={data['k']})")
    t.add_column("#", justify="right", style="dim")
    t.add_column("Width", justify="right", style="cyan")
    t.add_column("Height", justify="right", style="magenta")
    t.add_column("Area", justify="right", style="green")
    for i, a in enumerate(data["anchors"], 1):
        t.add_row(str(i), str(a["width"]), str(a["height"]), str(round(a["width"] * a["height"], 6)))
    console.print(t)


# ---------------------------------------------------------------------------
# Main render dispatcher
# ---------------------------------------------------------------------------

_RENDERERS = {
    "images_per_class": _render_images_per_class,
    "bbox_count_per_image": _render_bbox_count_per_image,
    "bbox_size_dist": _render_bbox_size_dist,
    "bbox_aspect_ratio": _render_bbox_aspect_ratio,
    "location_heatmaps": _render_location_heatmaps,
    "image_resolution_dist": _render_image_resolution_dist,
    "label_density": _render_label_density,
    "co_occurrence_matrix": _render_co_occurrence_matrix,
    "annotation_completeness": _render_annotation_completeness,
    "duplicate_detection": _render_duplicate_detection,
    "label_imbalance": _render_label_imbalance,
    "outlier_detection": _render_outlier_detection,
    "anchor_analysis": _render_anchor_analysis,
}


def render_cli(dataset_path: str, options: dict = None):
    """Parse the dataset, compute analytics, and render to the terminal."""
    console = Console()
    console.print(f"[bold green]Parsing dataset from[/bold green] {dataset_path}...")

    dataset = parse_yolo_dataset(dataset_path)
    analytics = compute_analytics(dataset, options=options)

    # Always render defaults
    _render_defaults(console, analytics)

    # Render optional analytics
    for key, renderer in _RENDERERS.items():
        if key in analytics:
            renderer(console, analytics[key])


def main():
    parser = argparse.ArgumentParser(
        description="YDVT — YOLO Dataset Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory (YOLO format)")
    parser.add_argument("--gui", action="store_true", help="Launch the Web GUI")
    parser.add_argument("--augment", action="store_true", help="Launch the interactive augmentation wizard")

    # Optional analytics flags
    analytics_group = parser.add_argument_group("optional analytics")
    analytics_group.add_argument("--all-analytics", action="store_true", help="Enable all optional analytics")
    analytics_group.add_argument("--images-per-class", action="store_true", help="Show unique images per class")
    analytics_group.add_argument("--bbox-count-per-image", action="store_true", help="Show bbox count stats per image")
    analytics_group.add_argument("--bbox-size-dist", action="store_true", help="Show bbox size distribution (small/medium/large)")
    analytics_group.add_argument("--bbox-aspect-ratio", action="store_true", help="Show bbox aspect ratio statistics")
    analytics_group.add_argument("--location-heatmaps", action="store_true", help="Show object location heatmap grid")
    analytics_group.add_argument("--image-resolution-dist", action="store_true", help="Show image resolution distribution")
    analytics_group.add_argument("--label-density", action="store_true", help="Show label density (avg bboxes/image)")
    analytics_group.add_argument("--co-occurrence-matrix", action="store_true", help="Show class co-occurrence matrix")
    analytics_group.add_argument("--annotation-completeness", action="store_true", help="Show annotation coverage stats")
    analytics_group.add_argument("--duplicate-detection", action="store_true", help="Detect duplicate images via MD5 hashing")
    analytics_group.add_argument("--label-imbalance", action="store_true", help="Show label imbalance metrics")
    analytics_group.add_argument("--outlier-detection", action="store_true", help="Detect outlier bounding boxes (>2σ)")
    analytics_group.add_argument("--anchor-analysis", action="store_true", help="Suggest anchor boxes via K-Means clustering")

    args = parser.parse_args()

    if args.gui:
        try:
            from ydvt.server import start_server
            start_server(args.dataset_path)
        except ImportError as e:
            print(f"Error starting server: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.augment:
        try:
            from ydvt.wizard import run_augmentation_wizard
            run_augmentation_wizard(args.dataset_path)
        except ImportError as e:
            print(f"Error starting augmentation wizard: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Build analytics options from CLI flags
        options = {}
        for flag_dest, option_key in _FLAG_TO_OPTION.items():
            cli_attr = flag_dest  # argparse converts hyphens to underscores
            if args.all_analytics or getattr(args, cli_attr, False):
                options[option_key] = True
        render_cli(args.dataset_path, options=options if options else None)


if __name__ == "__main__":
    main()
