"""Command-line interface for the autostereogram generator."""

from __future__ import annotations

import os
import warnings

# Suppress warnings via environment variable (applies to all subprocesses)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Aggressively suppress all warnings for clean UX
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

from huggingface_hub import scan_cache_dir
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .depth_estimator import DEFAULT_MODEL_ID, DEFAULT_PADDING, estimate_depth_map
from .stereogram import (
    DEFAULT_HUE_RANGE,
    DEFAULT_PATTERN_TYPE,
    DEFAULT_PERLIN_OCTAVES,
    DEFAULT_PERLIN_SCALE,
    DEFAULT_SATURATION_RANGE,
    DEFAULT_VALUE_RANGE,
    generate_autostereogram,
)
from .utils import validate_image_format

console = Console()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an autostereogram from an input image."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output stereogram image",
    )
    parser.add_argument(
        "--noise-width",
        type=int,
        default=None,
        help="Noise pattern width in pixels (default: auto)",
    )
    parser.add_argument(
        "--shift-range",
        type=int,
        default=None,
        help="Maximum pixel shift for depth mapping (default: auto)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override device selection (cpu/cuda)",
    )
    parser.add_argument(
        "--save-depth-map",
        default=None,
        help="Optional path to save the intermediate depth map",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached model weights and force re-download",
    )
    parser.add_argument(
        "--pattern-type",
        choices=["static", "perlin"],
        default=DEFAULT_PATTERN_TYPE,
        help=f"Pattern type for stereogram (default: {DEFAULT_PATTERN_TYPE})",
    )
    parser.add_argument(
        "--perlin-scale",
        type=float,
        default=DEFAULT_PERLIN_SCALE,
        help=f"Perlin noise scale factor - lower values create smoother waves (default: {DEFAULT_PERLIN_SCALE})",
    )
    parser.add_argument(
        "--perlin-octaves",
        type=int,
        default=DEFAULT_PERLIN_OCTAVES,
        help=f"Perlin noise octaves - number of noise layers (default: {DEFAULT_PERLIN_OCTAVES})",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Generate grayscale (monochrome) stereogram instead of color (color is default for Perlin patterns)",
    )
    parser.add_argument(
        "--hue-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=f"Hue range [0, 1] for color patterns (default: {DEFAULT_HUE_RANGE[0]:.1f} {DEFAULT_HUE_RANGE[1]:.1f})",
    )
    parser.add_argument(
        "--saturation-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=f"Saturation range [0, 1] for color patterns (default: {DEFAULT_SATURATION_RANGE[0]:.1f} {DEFAULT_SATURATION_RANGE[1]:.1f})",
    )
    parser.add_argument(
        "--value-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=f"Value (brightness) range [0, 1] for color patterns (default: {DEFAULT_VALUE_RANGE[0]:.1f} {DEFAULT_VALUE_RANGE[1]:.1f})",
    )
    parser.add_argument(
        "--hue-scale",
        type=float,
        default=None,
        help="Separate Perlin scale for hue channel (default: same as --perlin-scale)",
    )
    parser.add_argument(
        "--saturation-scale",
        type=float,
        default=None,
        help="Separate Perlin scale for saturation channel (default: same as --perlin-scale)",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=None,
        help="Separate Perlin scale for value channel (default: same as --perlin-scale)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING,
        help=f"Padding size in pixels to add around depth map (default: {DEFAULT_PADDING})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)",
    )
    return parser


def clear_model_cache(model_id: str) -> None:
    """
    Clear the HuggingFace cache for the specified model.
    
    Args:
        model_id: The HuggingFace model identifier (e.g., "depth-anything/Depth-Anything-V2-Base-hf")
    """
    cache_info = scan_cache_dir()
    # Find and delete revisions for the specified model
    for repo in cache_info.repos:
        if repo.repo_id == model_id:
            for revision in repo.revisions:
                revision.delete()


def _validate_positive_int(value: int | None, label: str) -> None:
    """
    Validate that a value is a positive integer if provided.
    
    Args:
        value: The value to validate (can be None)
        label: A label for the value to use in error messages
        
    Raises:
        ValueError: If value is not None and is not positive
    """
    if value is not None and value <= 0:
        raise ValueError(f"{label} must be a positive integer.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        console.print(f"[red]Error:[/red] input file not found: {input_path}")
        return 1

    if not validate_image_format(input_path):
        console.print(
            "[red]Error:[/red] unsupported input file format. "
            "The file format may not be supported by PIL/Pillow, or the file may be corrupted."
        )
        return 1

    try:
        _validate_positive_int(args.noise_width, "Noise width")
        _validate_positive_int(args.shift_range, "Shift range")
        if args.padding < 0:
            raise ValueError("Padding must be non-negative.")
        if args.perlin_scale <= 0:
            raise ValueError("Perlin scale must be positive.")
        if args.perlin_octaves < 1:
            raise ValueError("Perlin octaves must be at least 1.")
        if args.hue_range is not None:
            if len(args.hue_range) != 2 or args.hue_range[0] < 0 or args.hue_range[1] > 1 or args.hue_range[0] >= args.hue_range[1]:
                raise ValueError("Hue range must be two values in [0, 1] with min < max.")
        if args.saturation_range is not None:
            if len(args.saturation_range) != 2 or args.saturation_range[0] < 0 or args.saturation_range[1] > 1 or args.saturation_range[0] >= args.saturation_range[1]:
                raise ValueError("Saturation range must be two values in [0, 1] with min < max.")
        if args.value_range is not None:
            if len(args.value_range) != 2 or args.value_range[0] < 0 or args.value_range[1] > 1 or args.value_range[0] >= args.value_range[1]:
                raise ValueError("Value range must be two values in [0, 1] with min < max.")
        if args.hue_scale is not None and args.hue_scale <= 0:
            raise ValueError("Hue scale must be positive.")
        if args.saturation_scale is not None and args.saturation_scale <= 0:
            raise ValueError("Saturation scale must be positive.")
        if args.value_scale is not None and args.value_scale <= 0:
            raise ValueError("Value scale must be positive.")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    if args.clear_cache:
        try:
            with console.status("[bold yellow]Clearing model cache..."):
                clear_model_cache(DEFAULT_MODEL_ID)
            console.print("[green]✓[/green] Model cache cleared")
        except OSError as exc:
            console.print(f"[red]Error clearing model cache:[/red] {exc}")
            return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth_map_path = Path(args.save_depth_map) if args.save_depth_map else None
    if depth_map_path is not None:
        depth_map_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            depth_map = estimate_depth_map(
                input_path,
                device=args.device,
                save_path=depth_map_path,
                console=console,
                progress=progress,
                padding=args.padding,
            )
        except Exception as exc:
            console.print(f"[red]Error generating depth map:[/red] {exc}")
            return 1

        try:
            task4 = progress.add_task("[cyan]Generating autostereogram...", total=None)
            # Convert range tuples if provided
            hue_range = tuple(args.hue_range) if args.hue_range is not None else None
            saturation_range = tuple(args.saturation_range) if args.saturation_range is not None else None
            value_range = tuple(args.value_range) if args.value_range is not None else None
            
            # Determine color setting: False if --mono, None otherwise (defaults to True for perlin)
            color_setting = False if args.mono else None
            
            stereogram = generate_autostereogram(
                depth_map,
                noise_width=args.noise_width,
                shift_range=args.shift_range,
                pattern_type=args.pattern_type,
                perlin_scale=args.perlin_scale,
                perlin_octaves=args.perlin_octaves,
                color=color_setting,
                hue_range=hue_range,
                saturation_range=saturation_range,
                value_range=value_range,
                hue_scale=args.hue_scale,
                saturation_scale=args.saturation_scale,
                value_scale=args.value_scale,
            )
            progress.update(task4, completed=True)
        except Exception as exc:
            console.print(f"[red]Error generating autostereogram:[/red] {exc}")
            return 1

        try:
            task5 = progress.add_task("[cyan]Saving output image...", total=None)
            # Determine image mode based on stereogram dimensions
            if stereogram.ndim == 3:
                output_image = Image.fromarray(stereogram, mode="RGB")
            else:
                output_image = Image.fromarray(stereogram, mode="L")
            output_image.save(output_path)
            progress.update(task5, completed=True)
        except Exception as exc:
            console.print(f"[red]Error saving output image:[/red] {exc}")
            return 1

    elapsed_time = time.time() - start_time
    console.print(f"[green]✓[/green] Saved autostereogram to [bold]{output_path}[/bold]")
    if depth_map_path:
        console.print(f"[green]✓[/green] Saved depth map to [bold]{depth_map_path}[/bold]")
    console.print(f"[dim]Finished in {elapsed_time:.1f}s[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
