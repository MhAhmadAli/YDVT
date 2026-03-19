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

    # ── Step 3.5: Strict Mode ────────────────────────────────────────────
    strict_filter = questionary.confirm(
        "Enable Strict Mode? (only use images where ALL bboxes are the target class)",
        default=False,
    ).ask()

    if strict_filter is None:
        console.print("[yellow]Aborted.[/yellow]")
        return

    # ── Step 4: Confirm ──────────────────────────────────────────────────
    class_names = [dataset.classes.get(c, f"Class {c}") for c in selected_classes]
    console.print()
    console.print(Panel(
        f"[bold]Classes:[/bold] {', '.join(class_names)}\n"
        f"[bold]Augmentations:[/bold] {', '.join(selected_augs)}\n"
        f"[bold]Images per class:[/bold] {num_images}\n"
        f"[bold]Strict Mode:[/bold] {'enabled' if strict_filter else 'disabled'}",
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
            strict_filter=strict_filter,
        )

    console.print(
        f"\n[bold green]✓[/bold green] Generated "
        f"[bold]{result['generated_count']}[/bold] augmented images."
    )
    if result.get("skipped_classes"):
        skipped_names = [dataset.classes.get(c, f"Class {c}") for c in result["skipped_classes"]]
        console.print(
            f"[yellow]⚠[/yellow] Skipped "
            f"[bold]{', '.join(skipped_names)}[/bold] — no images with exclusively "
            "these classes found.\n"
            "[dim]Disable Strict Mode to include mixed-class images.[/dim]"
        )

def run_headless_augmentation(dataset_path: str, class_names: list, aug_names: list, num_images: int, strict_filter: bool) -> None:
    """Execute augmentation headlessly from CLI arguments."""
    console.print(f"\n[bold green]Parsing dataset from[/bold green] {dataset_path}…")
    dataset = parse_yolo_dataset(dataset_path)
    analytics = compute_analytics(dataset)

    dist = analytics["class_distribution"]
    if not dist:
        console.print("[red]No classes found in the dataset. Aborting.[/red]")
        sys.exit(1)

    # 1. Resolve classes
    selected_classes = []
    for cname in class_names:
        class_id = None
        for cid, name in dataset.classes.items():
            if name == cname or str(cid) == cname:
                class_id = cid
                break
        if class_id is None:
            console.print(f"[red]Error: Class '{cname}' not found in dataset.[/red]")
            sys.exit(1)
        selected_classes.append(class_id)
        
    # Remove duplicates while preserving list order
    selected_classes = list(dict.fromkeys(selected_classes))

    # 2. Validate augmentations
    available_augs = {a["name"] for a in list_available_augmentations()}
    for aug in aug_names:
        if aug not in available_augs:
            console.print(f"[red]Error: Unknown augmentation '{aug}'. Available options: {', '.join(sorted(available_augs))}[/red]")
            sys.exit(1)

    # 3. Print Summary
    resolved_class_names = [dataset.classes.get(c, f"Class {c}") for c in selected_classes]
    console.print()
    console.print(Panel(
        f"[bold]Classes:[/bold] {', '.join(resolved_class_names)}\n"
        f"[bold]Augmentations:[/bold] {', '.join(aug_names)}\n"
        f"[bold]Images per class:[/bold] {num_images}\n"
        f"[bold]Strict Mode:[/bold] {'enabled' if strict_filter else 'disabled'}",
        title="[bold blue]Headless Augmentation Summary[/bold blue]",
        expand=False,
    ))

    # 4. Execute
    with console.status("[bold cyan]Generating augmented images…[/bold cyan]"):
        try:
            result = apply_augmentations(
                dataset=dataset,
                target_classes=selected_classes,
                augmentation_names=aug_names,
                num_images=num_images,
                strict_filter=strict_filter,
            )
        except Exception as e:
            console.print(f"[red]Error during augmentation: {e}[/red]")
            sys.exit(1)

    console.print(
        f"\n[bold green]✓[/bold green] Generated "
        f"[bold]{result['generated_count']}[/bold] augmented images."
    )
    if result.get("skipped_classes"):
        skipped_names = [dataset.classes.get(c, f"Class {c}") for c in result["skipped_classes"]]
        console.print(
            f"[yellow]⚠[/yellow] Skipped "
            f"[bold]{', '.join(skipped_names)}[/bold] — no images with exclusively "
            "these classes found.\n"
            "[dim]Disable Strict Mode to include mixed-class images.[/dim]"
        )
