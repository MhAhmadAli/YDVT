import argparse
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ydvt.parser import parse_yolo_dataset
from ydvt.analytics import compute_analytics

def render_cli(dataset_path: str):
    console = Console()
    console.print(f"[bold green]Parsing dataset from[/bold green] {dataset_path}...")
    
    dataset = parse_yolo_dataset(dataset_path)
    analytics = compute_analytics(dataset)
    
    summary = analytics["summary"]
    console.print()
    console.print(Panel(
        f"Total Images: {summary['total_images']}\n"
        f"Images w/ Annotations: {summary['images_with_annotations']}\n"
        f"Total BBoxes: {summary['total_bboxes']}\n"
        f"Unmapped Classes: {summary['unmapped_classes']}", 
        title="[bold blue]Dataset Summary[/bold blue]",
        expand=False
    ))

    table = Table(title="Class Distribution")
    table.add_column("Class", justify="left", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Avg BBox Size (W x H)", justify="right", style="green")

    dist = analytics["class_distribution"]
    avg_sizes = analytics["avg_bbox_sizes"]
    
    for cls_name, count in dist.items():
        if isinstance(cls_name, int) and summary["unmapped_classes"]:
            name = f"Class {cls_name} (Missing from classes.txt)"
        else:
            name = str(cls_name)

        size = avg_sizes.get(cls_name)
        if not size and cls_name not in avg_sizes:
            # Fallback for dynamic unmapped classes key mismatch
            for k in avg_sizes.keys():
                if str(k) == str(cls_name) or (isinstance(cls_name, str) and cls_name.endswith(str(k))):
                    size = avg_sizes[k]
                    break
                    
        if size:
            size_str = f"{size['w']:.3f} x {size['h']:.3f}"
        else:
            size_str = "0.000 x 0.000"
            
        table.add_row(name, str(count), size_str)

    console.print(table)

    # Split distribution table
    split_dist = analytics.get("split_distribution", {})
    if split_dist:
        split_table = Table(title="Split Distribution")
        split_table.add_column("Split", justify="left", style="cyan", no_wrap=True)
        split_table.add_column("Images", justify="right", style="magenta")
        split_table.add_column("BBoxes", justify="right", style="green")
        split_table.add_column("%", justify="right", style="yellow")

        for split_name, info in split_dist.items():
            split_table.add_row(
                split_name.capitalize(),
                str(info["images"]),
                str(info["bboxes"]),
                f"{info['percentage']}%",
            )

        console.print(split_table)

def main():
    parser = argparse.ArgumentParser(description="Image Dataset Visualization Tool")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory (YOLO format)")
    parser.add_argument("--gui", action="store_true", help="Launch the Web GUI")
    parser.add_argument("--augment", action="store_true", help="Launch the interactive augmentation wizard")
    
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
        render_cli(args.dataset_path)

if __name__ == "__main__":
    main()
