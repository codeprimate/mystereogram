# Mystereogram

A Python command-line tool that generates Magic Eye-style autostereograms from 2D images using AI depth estimation.

## Features

- **AI-Powered Depth Estimation**: Uses DepthAnything V2-base model to automatically extract depth information from images
- **Autostereogram Generation**: Creates noise-pattern-based autostereograms that reveal 3D depth when viewed with the "Magic Eye" technique
- **Multiple Pattern Types**: Choose between static random noise or smooth Perlin noise patterns
- **Color Control**: Full control over color generation with customizable hue, saturation, and brightness ranges
- **Grayscale Mode**: Option to generate monochrome stereograms
- **GPU Acceleration**: Supports CPU and CUDA (NVIDIA GPUs) for faster processing
- **Automatic Optimization**: Auto-calculates optimal noise pattern width and shift range based on image size
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install from Source

1. Clone the repository:
```bash
git clone <repository-url>
cd mystereogram
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```


## Usage

### Basic Usage

```bash
mystereogram -i input.jpg -o output.png
```

Or with long-form flags:

```bash
mystereogram --input input.jpg --output output.png
```

### Command-Line Options

**Required Arguments:**
- `-i, --input INPUT`: Path to the input image file
- `-o, --output OUTPUT`: Path to the output stereogram image

**Basic Options:**
- `--noise-width WIDTH`: Width of noise pattern in pixels (default: auto-calculated based on image size)
- `--shift-range RANGE`: Maximum pixel shift for depth mapping (default: auto-calculated based on image size)
- `--device DEVICE`: Override device selection (`cpu` or `cuda`). Default: auto-detect
- `--save-depth-map PATH`: Save the intermediate depth map to the specified path
- `--clear-cache`: Clear cached model weights and force re-download
- `-h, --help`: Show help message

**Pattern Options:**
- `--pattern-type TYPE`: Pattern type for stereogram - `static` (random noise) or `perlin` (smooth Perlin noise waves). Default: `perlin`
- `--perlin-scale SCALE`: Perlin noise scale factor - lower values create smoother waves (default: 0.2)
- `--perlin-octaves OCTAVES`: Perlin noise octaves - number of noise layers (default: 6)
- `--mono`: Generate grayscale (monochrome) stereogram instead of color. Color is default for Perlin patterns

**Color Options (for Perlin patterns):**
- `--hue-range MIN MAX`: Hue range [0, 1] for color patterns (default: 0.0 1.0 - full spectrum)
- `--saturation-range MIN MAX`: Saturation range [0, 1] for color patterns (default: 0.7 1.0 - high saturation)
- `--value-range MIN MAX`: Value (brightness) range [0, 1] for color patterns (default: 0.7 1.0 - high brightness)
- `--hue-scale SCALE`: Separate Perlin scale for hue channel (default: same as `--perlin-scale`)
- `--saturation-scale SCALE`: Separate Perlin scale for saturation channel (default: same as `--perlin-scale`)
- `--value-scale SCALE`: Separate Perlin scale for value channel (default: same as `--perlin-scale`)

### Examples

**Basic stereogram generation:**
```bash
mystereogram -i photo.jpg -o stereogram.png
```

**With custom noise pattern width:**
```bash
mystereogram -i photo.jpg -o stereogram.png --noise-width 48
```

**Generate grayscale stereogram:**
```bash
mystereogram -i photo.jpg -o stereogram.png --mono
```

**Use static random noise pattern instead of Perlin:**
```bash
mystereogram -i photo.jpg -o stereogram.png --pattern-type static
```

**Custom Perlin noise settings:**
```bash
mystereogram -i photo.jpg -o stereogram.png --perlin-scale 0.15 --perlin-octaves 8
```

**Custom color range (warm colors only):**
```bash
mystereogram -i photo.jpg -o stereogram.png --hue-range 0.0 0.2 --saturation-range 0.8 1.0
```

**Save depth map for debugging:**
```bash
mystereogram -i photo.jpg -o stereogram.png --save-depth-map depth.png
```

**Force CPU processing:**
```bash
mystereogram -i photo.jpg -o stereogram.png --device cpu
```

**Clear model cache:**
```bash
mystereogram --clear-cache
```

## How It Works

1. **Input Processing**: The input image is validated and automatically resized to exactly 1 megapixel (maintaining aspect ratio) for efficient processing.

2. **Depth Estimation**: The DepthAnything V2-base model analyzes the image and generates a depth map, estimating which parts of the image are closer or farther away.

3. **Stereogram Generation**: A random noise pattern is created, and pixels are horizontally shifted based on the depth map values. When viewed with the "Magic Eye" technique (relaxing your eyes to look "through" the image), the depth information creates a 3D effect.

## Viewing Autostereograms

To see the 3D effect in the generated autostereogram:

1. Hold the image close to your face
2. Relax your eyes and look "through" the image (as if focusing on something behind it)
3. Slowly move the image away from your face while maintaining the relaxed focus
4. The 3D depth should become visible

This technique is the same as viewing traditional "Magic Eye" images.

## Supported Image Formats

**Input formats:** JPEG, PNG, BMP, TIFF

**Output formats:** JPEG, PNG (determined by output file extension)

## Performance

- **First Run**: The model (~390MB) will be downloaded automatically. This only happens once.
- **Processing Time**: 
  - Depth estimation: 5-30 seconds (depending on image size and hardware)
  - Stereogram generation: < 5 seconds for typical images
- **GPU Acceleration**: Significantly faster on NVIDIA GPUs (CUDA)

## Troubleshooting

**Model download fails:**
- Check your internet connection
- The model is cached after first download, so subsequent runs work offline
- Use `--clear-cache` to force re-download if the cache is corrupted

**Out of memory errors:**
- Large images are automatically resized to 1MP
- If issues persist, try processing smaller images or use `--device cpu`

**Poor stereogram quality:**
- Try adjusting `--noise-width` (typically 32-64 pixels)
- Try adjusting `--shift-range` (typically 5-20 pixels)
- Try different pattern types: `--pattern-type static` for traditional random noise, or `--pattern-type perlin` for smoother waves
- Adjust Perlin noise with `--perlin-scale` (lower = smoother) and `--perlin-octaves` (more = more detail)
- Images with clear depth variation work best


## Development

### Running Tests

```bash
pytest
```

### Project Structure

```
mystereogram/
├── stereogram_generator/
│   ├── cli.py              # Command-line interface
│   ├── depth_estimator.py  # Depth estimation using DepthAnything
│   ├── stereogram.py       # Autostereogram generation
│   └── utils.py            # Utility functions
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
└── setup.py               # Package configuration
```

## License

[Add your license information here]

## Acknowledgments

- Uses the [DepthAnything V2-base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) model for depth estimation
- Built with PyTorch, Transformers, and Pillow

