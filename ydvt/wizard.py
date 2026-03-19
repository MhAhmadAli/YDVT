"""
CLI Augmentation Wizard for YDVT.

Provides an interactive terminal wizard that guides the user through
selecting target classes, choosing augmentations, and specifying the
number of images to generate — then executes the augmentation pipeline.
"""

import sys

import questionary
from rich.console import Console
from rich.panel import Panel

from ydvt.parser import parse_yolo_dataset
from ydvt.analytics import compute_analytics
from ydvt.augmenter import apply_augmentations, list_available_augmentations


console = Console()


def run_augmentation_wizard(dataset_path: str) -> None:
    """Launch the interactive augmentation wizard."""

    console.print(f"\n[bold green]Parsing dataset from[/bold green] {dataset_path}…")
    dataset = parse_yolo_dataset(dataset_path)
    analytics = compute_analytics(dataset)

    dist = analytics["class_distribution"]
    if not dist:
        console.print("[red]No classes found in the dataset. Aborting.[/red]")
        sys.exit(1)

    # ── Step 1: Select target classes ────────────────────────────────────
    class_choices = []
    for class_name, count in dist.items():
        # Resolve class id from the dataset
        class_id = None
        for cid, cname in dataset.classes.items():
            if cname == class_name or f"Class {cid}" == class_name:
                class_id = cid
                break
        label = f"{class_name}  ({count} instances)"
        class_choices.append(questionary.Choice(title=label, value=class_id))

    selected_classes = questionary.checkbox(
        "Select target classes to augment:",
        choices=class_choices,
    ).ask()

    if not selected_classes:
        console.print("[yellow]No classes selected. Aborting.[/yellow]")
        return

    # ── Step 2: Select augmentations ─────────────────────────────────────
    aug_list = list_available_augmentations()
    aug_choices = [
        questionary.Choice(
            title=f"{a['label']}  — {a['description']}",
            value=a["name"],
        )
        for a in aug_list
    ]

    selected_augs = questionary.checkbox(
        "Select augmentations to apply:",
        choices=aug_choices,
    ).ask()

    if not selected_augs:
        console.print("[yellow]No augmentations selected. Aborting.[/yellow]")
        return

    # ── Step 3: Number of images per class ───────────────────────────────
    num_str = questionary.text(
        "Number of augmented images per class:",
        default="5",
        validate=lambda val: val.isdigit() and int(val) > 0 or "Enter a positive integer",
    ).ask()

    if num_str is None:
        console.print("[yellow]Aborted.[/yellow]")
        return

    num_images = int(num_str)

    # ── Step 4: Confirm ──────────────────────────────────────────────────
    class_names = [dataset.classes.get(c, f"Class {c}") for c in selected_classes]
    console.print()
    console.print(Panel(
        f"[bold]Classes:[/bold] {', '.join(class_names)}\n"
        f"[bold]Augmentations:[/bold] {', '.join(selected_augs)}\n"
        f"[bold]Images per class:[/bold] {num_images}",
        title="[bold blue]Augmentation Summary[/bold blue]",
        expand=False,
    ))

    confirm = questionary.confirm("Proceed with augmentation?", default=True).ask()
    if not confirm:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # ── Step 5: Execute ──────────────────────────────────────────────────
    with console.status("[bold cyan]Generating augmented images…[/bold cyan]"):
        result = apply_augmentations(
            dataset=dataset,
            target_classes=selected_classes,
            augmentation_names=selected_augs,
            num_images=num_images,
        )

    console.print(
        f"\n[bold green]✓[/bold green] Generated "
        f"[bold]{result['generated_count']}[/bold] augmented images."
    )
