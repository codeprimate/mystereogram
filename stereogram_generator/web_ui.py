"""Web interface for the autostereogram generator using Gradio."""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import gradio as gr

# Set up logging
logger = logging.getLogger(__name__)

from .depth_estimator import (
    DEFAULT_PADDING,
    load_depth_model,
    postprocess_depth_map,
    _estimate_depth,
    pad_depth_map,
    INVERT_DEPTH,
    invert_depth_map,
)
from .stereogram import (
    DEFAULT_HUE_RANGE,
    DEFAULT_PATTERN_TYPE,
    DEFAULT_PERLIN_OCTAVES,
    DEFAULT_PERLIN_SCALE,
    DEFAULT_SATURATION_RANGE,
    DEFAULT_VALUE_RANGE,
    generate_autostereogram,
)
from .utils import get_device, normalize_image, resize_image


# Global model cache to avoid reloading
_model_cache: dict[str, tuple] = {}


@dataclass
class StereogramConfig:
    """Configuration for stereogram generation parameters."""
    pattern_type: str
    mono: bool
    perlin_scale: float
    perlin_octaves: int
    hue_range_min: float
    hue_range_max: float
    saturation_range_min: float
    saturation_range_max: float
    value_range_min: float
    value_range_max: float
    hue_scale: Optional[float]
    saturation_scale: Optional[float]
    value_scale: Optional[float]
    noise_width: Optional[int]
    shift_range: Optional[int]


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    device: str
    padding: int
    show_depth_map: bool
    input_filename: Optional[str] = None


def get_cached_model(device: str):
    """Get or load the depth estimation model for the specified device."""
    if device not in _model_cache:
        model, processor = load_depth_model(device=device)
        _model_cache[device] = (model, processor)
    return _model_cache[device]


def process_image_for_web(
    image_path: str,
    stereogram_config: StereogramConfig,
    processing_config: ProcessingConfig,
) -> Tuple[Optional[str], Optional[Image.Image], str]:
    """
    Process an image and generate a stereogram.
    
    Args:
        image_path: Path to the input image file
        stereogram_config: Configuration for stereogram generation
        processing_config: Configuration for image processing
    
    Returns:
        Tuple of (stereogram_file_path, depth_map_image, info_text)
    """
    if image_path is None:
        return None, None, "Please upload an image"
    
    try:
        start_time = time.time()
        
        # Load image from file path
        image = Image.open(image_path)
        
        # Log all UI parameters received
        logger.info("=" * 60)
        logger.info("GENERATION START - UI Parameters Received:")
        logger.info(f"  pattern_type: {stereogram_config.pattern_type!r}")
        logger.info(f"  mono: {stereogram_config.mono}")
        logger.info(f"  device: {processing_config.device!r}")
        logger.info(f"  perlin_scale: {stereogram_config.perlin_scale}")
        logger.info(f"  perlin_octaves: {stereogram_config.perlin_octaves}")
        logger.info(f"  hue_range: ({stereogram_config.hue_range_min}, {stereogram_config.hue_range_max})")
        logger.info(f"  saturation_range: ({stereogram_config.saturation_range_min}, {stereogram_config.saturation_range_max})")
        logger.info(f"  value_range: ({stereogram_config.value_range_min}, {stereogram_config.value_range_max})")
        logger.info(f"  hue_scale: {stereogram_config.hue_scale}")
        logger.info(f"  saturation_scale: {stereogram_config.saturation_scale}")
        logger.info(f"  value_scale: {stereogram_config.value_scale}")
        logger.info(f"  noise_width: {stereogram_config.noise_width}")
        logger.info(f"  shift_range: {stereogram_config.shift_range}")
        logger.info(f"  padding: {processing_config.padding}")
        logger.info(f"  show_depth_map: {processing_config.show_depth_map}")
        logger.info("=" * 60)
        
        # Determine device
        if processing_config.device == "auto":
            selected_device = get_device()
        else:
            selected_device = processing_config.device
        
        logger.info(f"Selected device: {selected_device}")
        
        # Normalize image to RGB format (handles all image modes)
        image = normalize_image(image)
        
        # Resize image to 1MP
        resized = resize_image(image)
        resized_size = resized.size
        
        # Load model
        model, processor = get_cached_model(selected_device)
        
        # Estimate depth
        depth_tensor = _estimate_depth(resized, model, processor, selected_device)
        depth_map = postprocess_depth_map(depth_tensor, resized_size)
        if INVERT_DEPTH:
            depth_map = invert_depth_map(depth_map)
        
        # Add padding
        depth_map = pad_depth_map(depth_map, padding=processing_config.padding)
        
        # Prepare color settings
        color_setting = False if stereogram_config.mono else None
        
        # Match CLI behavior: pass None for defaults, tuples only when explicitly set
        # This ensures web UI matches CLI when using default values
        if stereogram_config.mono:
            hue_range = None
            saturation_range = None
            value_range = None
        else:
            # Check if values match defaults - if so, pass None (like CLI does)
            hue_range = None if (stereogram_config.hue_range_min, stereogram_config.hue_range_max) == DEFAULT_HUE_RANGE else (stereogram_config.hue_range_min, stereogram_config.hue_range_max)
            saturation_range = None if (stereogram_config.saturation_range_min, stereogram_config.saturation_range_max) == DEFAULT_SATURATION_RANGE else (stereogram_config.saturation_range_min, stereogram_config.saturation_range_max)
            value_range = None if (stereogram_config.value_range_min, stereogram_config.value_range_max) == DEFAULT_VALUE_RANGE else (stereogram_config.value_range_min, stereogram_config.value_range_max)
        
        logger.info("Prepared parameters for generate_autostereogram:")
        logger.info(f"  pattern_type: {stereogram_config.pattern_type!r}")
        logger.info(f"  color_setting: {color_setting}")
        logger.info(f"  hue_range: {hue_range}")
        logger.info(f"  saturation_range: {saturation_range}")
        logger.info(f"  value_range: {value_range}")
        logger.info(f"  perlin_scale: {stereogram_config.perlin_scale}")
        logger.info(f"  perlin_octaves: {stereogram_config.perlin_octaves}")
        logger.info(f"  hue_scale: {stereogram_config.hue_scale}")
        logger.info(f"  saturation_scale: {stereogram_config.saturation_scale}")
        logger.info(f"  value_scale: {stereogram_config.value_scale}")
        logger.info(f"  noise_width: {stereogram_config.noise_width}")
        logger.info(f"  shift_range: {stereogram_config.shift_range}")
        
        # Generate stereogram
        logger.info("Calling generate_autostereogram...")
        stereogram_array = generate_autostereogram(
            depth_map,
            noise_width=stereogram_config.noise_width,
            shift_range=stereogram_config.shift_range,
            pattern_type=stereogram_config.pattern_type,
            perlin_scale=stereogram_config.perlin_scale,
            perlin_octaves=stereogram_config.perlin_octaves,
            color=color_setting,
            hue_range=hue_range,
            saturation_range=saturation_range,
            value_range=value_range,
            hue_scale=stereogram_config.hue_scale,
            saturation_scale=stereogram_config.saturation_scale,
            value_scale=stereogram_config.value_scale,
        )
        logger.info(f"generate_autostereogram completed. Output shape: {stereogram_array.shape}, dtype: {stereogram_array.dtype}")
        
        # Convert to PIL Image
        if stereogram_array.ndim == 3:
            stereogram_image = Image.fromarray(stereogram_array, mode="RGB")
        else:
            stereogram_image = Image.fromarray(stereogram_array, mode="L")
        
        # Generate output filename based on input filename and timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        if processing_config.input_filename:
            # Extract base name without extension
            input_path = Path(processing_config.input_filename)
            base_name = input_path.stem
            extension = input_path.suffix or ".png"
        else:
            base_name = "stereogram"
            extension = ".png"
        
        output_filename = f"{base_name}_{timestamp}{extension}"
        
        # Save to temporary file with custom filename
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / output_filename
        stereogram_image.save(str(output_path))
        
        # Create depth map visualization if requested
        depth_map_image = None
        if processing_config.show_depth_map:
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_map_image = Image.fromarray(depth_vis, mode="L")
        
        elapsed_time = time.time() - start_time
        
        # Create info text
        info_lines = [
            f"Processing time: {elapsed_time:.1f}s",
            f"Output size: {stereogram_image.size[0]}×{stereogram_image.size[1]}",
            f"Device: {selected_device}",
        ]
        info_text = "\n".join(info_lines)
        
        return str(output_path), depth_map_image, info_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, None, error_msg


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="My Stereogram Generator") as interface:
        gr.Markdown(
            "# My Stereogram Generator\n"
            "Transform images into autostereograms using AI depth estimation."
        )
        
        with gr.Row():
            # Left column: Input and Output
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="filepath",
                    height=300,
                )
                
                generate_btn = gr.Button(
                    "Generate Stereogram",
                    variant="primary",
                    size="lg",
                )
                
                with gr.Row():
                    stereogram_output = gr.Image(
                        label="Generated Stereogram",
                        type="filepath",
                        height=400,
                        interactive=False,
                    )
                    
                    depth_map_output = gr.Image(
                        label="Depth Map",
                        type="pil",
                        height=400,
                        interactive=False,
                        visible=True,
                    )
                
                info_text = gr.Textbox(
                    label="Info",
                    value="",
                    interactive=False,
                    lines=4,
                )
            
            # Right column: Controls (Tabs)
            with gr.Column(scale=1):
                with gr.Tabs():
                    # Basic Tab
                    with gr.Tab("Basic"):
                        pattern_type = gr.Radio(
                            ["static", "perlin"],
                            value=DEFAULT_PATTERN_TYPE,
                            label="Pattern Type",
                            info="Static: random noise | Perlin: smooth waves",
                        )
                        mono = gr.Checkbox(
                            label="Grayscale",
                            value=False,
                            info="Generate monochrome stereogram",
                        )
                        device = gr.Radio(
                            ["auto", "cpu", "cuda"],
                            value="auto",
                            label="Device",
                            info="Auto: use GPU if available",
                        )
                    
                    # Pattern Tab
                    with gr.Tab("Pattern"):
                        perlin_scale = gr.Slider(
                            0.01,
                            1.0,
                            DEFAULT_PERLIN_SCALE,
                            step=0.01,
                            label="Smoothness (Perlin Scale)",
                            info="Lower = smoother waves",
                        )
                        perlin_octaves = gr.Slider(
                            1,
                            10,
                            DEFAULT_PERLIN_OCTAVES,
                            step=1,
                            label="Detail Level (Octaves)",
                            info="More = more detail",
                        )
                    
                    # Color Tab (conditional)
                    with gr.Tab("Color", visible=True) as color_tab:
                        hue_min = gr.Slider(
                            0.0,
                            1.0,
                            DEFAULT_HUE_RANGE[0],
                            step=0.01,
                            label="Hue Min",
                            info="0.0 = red, 0.33 = green, 0.67 = blue",
                        )
                        hue_max = gr.Slider(
                            0.0,
                            1.0,
                            DEFAULT_HUE_RANGE[1],
                            step=0.01,
                            label="Hue Max",
                        )
                        saturation_min = gr.Slider(
                            0.0,
                            1.0,
                            DEFAULT_SATURATION_RANGE[0],
                            step=0.01,
                            label="Saturation Min",
                            info="0.0 = grayscale, 1.0 = full color",
                        )
                        saturation_max = gr.Slider(
                            0.0,
                            1.0,
                            DEFAULT_SATURATION_RANGE[1],
                            step=0.01,
                            label="Saturation Max",
                        )
                        value_min = gr.Slider(
                            0.0,
                            1.0,
                            DEFAULT_VALUE_RANGE[0],
                            step=0.01,
                            label="Value (Brightness) Min",
                            info="0.0 = black, 1.0 = full brightness",
                        )
                        value_max = gr.Slider(
                            0.0,
                            1.0,
                            DEFAULT_VALUE_RANGE[1],
                            step=0.01,
                            label="Value (Brightness) Max",
                        )
                        hue_scale = gr.Slider(
                            0.01,
                            1.0,
                            None,
                            step=0.01,
                            label="Hue Scale (Optional)",
                            info="Leave empty to use Perlin Scale",
                        )
                        saturation_scale = gr.Slider(
                            0.01,
                            1.0,
                            None,
                            step=0.01,
                            label="Saturation Scale (Optional)",
                            info="Leave empty to use Perlin Scale",
                        )
                        value_scale = gr.Slider(
                            0.01,
                            1.0,
                            None,
                            step=0.01,
                            label="Value Scale (Optional)",
                            info="Leave empty to use Perlin Scale",
                        )
                    
                    # Advanced Tab
                    with gr.Tab("Advanced"):
                        noise_width = gr.Number(
                            label="Noise Width",
                            value=None,
                            info="Leave empty for auto (recommended)",
                            precision=0,
                        )
                        shift_range = gr.Number(
                            label="Shift Range",
                            value=None,
                            info="Leave empty for auto (recommended)",
                            precision=0,
                        )
                        padding = gr.Number(
                            label="Padding",
                            value=DEFAULT_PADDING,
                            info="Padding around depth map in pixels",
                            precision=0,
                        )
        
        # Event handlers
        def update_color_tab_visibility(pattern_type_val: str, mono_val: bool):
            """Update Color tab visibility based on pattern type and mono setting."""
            visible = pattern_type_val == "perlin" and not mono_val
            return gr.update(visible=visible)
        
        pattern_type.change(
            fn=update_color_tab_visibility,
            inputs=[pattern_type, mono],
            outputs=[color_tab],
        )
        mono.change(
            fn=update_color_tab_visibility,
            inputs=[pattern_type, mono],
            outputs=[color_tab],
        )
        
        def generate_handler(
            img,
            pattern_type_val,
            mono_val,
            device_val,
            perlin_scale_val,
            perlin_octaves_val,
            hue_min_val,
            hue_max_val,
            sat_min_val,
            sat_max_val,
            val_min_val,
            val_max_val,
            hue_scale_val,
            sat_scale_val,
            val_scale_val,
            noise_width_val,
            shift_range_val,
            padding_val,
        ):
            """Handle the generate button click."""
            # Extract filename from image path
            input_filename = None
            if img:
                input_filename = Path(img).name
            
            # Convert scale values: if None or exactly the minimum (0.01), treat as None
            # Gradio sliders with value=None and minimum=0.01 may return 0.01 instead of None
            SCALE_MIN = 0.01
            hue_scale_processed = None if (hue_scale_val is None or hue_scale_val == SCALE_MIN) else hue_scale_val
            sat_scale_processed = None if (sat_scale_val is None or sat_scale_val == SCALE_MIN) else sat_scale_val
            val_scale_processed = None if (val_scale_val is None or val_scale_val == SCALE_MIN) else val_scale_val
            
            # Create configuration objects
            stereogram_config = StereogramConfig(
                pattern_type=pattern_type_val,
                mono=mono_val,
                perlin_scale=perlin_scale_val,
                perlin_octaves=perlin_octaves_val,
                hue_range_min=hue_min_val,
                hue_range_max=hue_max_val,
                saturation_range_min=sat_min_val,
                saturation_range_max=sat_max_val,
                value_range_min=val_min_val,
                value_range_max=val_max_val,
                hue_scale=hue_scale_processed,
                saturation_scale=sat_scale_processed,
                value_scale=val_scale_processed,
                noise_width=int(noise_width_val) if noise_width_val is not None and noise_width_val > 0 else None,
                shift_range=int(shift_range_val) if shift_range_val is not None and shift_range_val > 0 else None,
            )
            
            processing_config = ProcessingConfig(
                device=device_val,
                padding=int(padding_val),
                show_depth_map=True,  # Always show depth map
                input_filename=input_filename,
            )
            
            # Always show depth map
            result = process_image_for_web(
                img,
                stereogram_config,
                processing_config,
            )
            
            stereogram, depth_map, info = result
            
            return (
                stereogram,
                depth_map,
                info,
            )
        
        generate_btn.click(
            fn=generate_handler,
            inputs=[
                input_image,
                pattern_type,
                mono,
                device,
                perlin_scale,
                perlin_octaves,
                hue_min,
                hue_max,
                saturation_min,
                saturation_max,
                value_min,
                value_max,
                hue_scale,
                saturation_scale,
                value_scale,
                noise_width,
                shift_range,
                padding,
            ],
            outputs=[stereogram_output, depth_map_output, info_text],
        )

            # Footer with copyright and GitHub link
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid var(--border-color-primary);">
                    <p style="margin: 10px 0;">
                        <a href="https://github.com/codeprimate/mystereogram" target="_blank" style="text-decoration: none; color: var(--link-text-color); display: inline-flex; align-items: center; gap: 8px;">
                            ©2026 codeprimate | View on GitHub
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: middle;">
                                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                            </svg>
                        </a>
                    </p>
                </div>
                """
            )
        
    
    return interface


def main() -> int:
    """Main entry point for mystereogram-web command."""
    parser = argparse.ArgumentParser(
        description="Launch web interface for mystereogram generator"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number for web server (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Validate port
    if not (1 <= args.port <= 65535):
        print(f"Error: Port must be between 1 and 65535")
        return 1
    
    # Create interface
    interface = create_interface()
    
    # Launch server
    try:
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=not args.no_browser,
            show_error=True,
            theme=gr.themes.Soft(),
        )
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"Error: Port {args.port} is already in use. Try a different port with --port")
        else:
            print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

