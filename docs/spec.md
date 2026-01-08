# Feature: Python CLI Autostereogram Generator

## Problem Statement

Create a command-line application that transforms a 2D image into a "Magic Eye" style autostereogram. The application should:
1. Accept an input image file
2. Compute a depth map from the image using DepthAnything
3. Generate a noise-pattern-based autostereogram from the depth map
4. Save the resulting autostereogram to a file

The application should be simple, well-structured, and maintainable, using Python's venv for dependency management.

## Requirements

### Functional Requirements

- **Input Processing**
  - Accept a single input image file path as a command-line argument
  - Support common image formats (JPEG, PNG, etc.)
  - Validate that the input file exists and is a valid image
  - Resize to exactly 1MP using bicubic

- **Depth Map Generation**
  - Use DepthAnything V2-base (`depth-anything/Depth-Anything-V2-Base-hf`) to compute depth maps from RGB images
  - Handle model loading and inference using HuggingFace `transformers` library
  - Interpolate depth map output to original image size using bicubic interpolation
  - Normalize depth map values to [0, 1] range using min-max normalization
  - Invert depth map if needed (hardcoded based on model output characteristics)
  - Scale normalized depth to shift range for stereogram generation
  - Support CPU and GPU (CUDA) inference
  - Automatically download and cache model weights on first use (~390MB)
  - Provide progress feedback during model download (via transformers/tqdm)
  - Handle model download failures gracefully with retry logic (built into huggingface_hub)

- **Autostereogram Generation**
  - Generate noise-based autostereograms (not pattern-image-based)
  - Create random grayscale noise pattern (typically 32-64 pixels wide, single column or narrow strip)
  - Apply horizontal pixel shifts based on depth map values (row-by-row processing)
  - For each pixel, calculate shift and copy from `(current_column - pattern_width + shift)` position
  - Ensure proper wrapping and continuity in the generated image
  - Maintain input image dimensions in output

- **Output**
  - Save the generated autostereogram to a specified output file path
  - Default output format: JPEG (JPG)
  - Support common output formats (JPEG, PNG) - format determined by output file extension
  - Provide progress feedback or completion message

- **CLI Interface**
  - Executable command-line interface (runs directly without `python -m`)
  - Command syntax: `mystereogram [OPTIONS] INPUT OUTPUT`
  - Required arguments:
    - `INPUT`: Path to input image file
    - `OUTPUT`: Path to output stereogram file
  - Optional arguments:
    - `--noise-width WIDTH`: Width of noise pattern in pixels (default: auto-calculated as `max(64, min(128, image_width // 15))` - typically 64-128 pixels, approximately 1/15 of image width)
    - `--shift-range RANGE`: Maximum pixel shift for depth mapping (default: auto-calculated as `max(20, min(60, image_width // 30))` - typically 20-60 pixels, approximately 1/30 of image width)
    - `--device DEVICE`: Override device selection (cpu/cuda, default: auto)
    - `--save-depth-map PATH`: Save intermediate depth map to specified path
    - `--clear-cache`: Clear model cache and force re-download
    - `--help`: Show help message
  - Help text explaining usage
  - Appropriate error messages for invalid inputs

### Technical Constraints

- **Python Environment**
  - Use Python 3.8+ (compatible with modern ML libraries)
  - Manage dependencies via `venv` (virtual environment)
  - Include `requirements.txt` for dependency specification

- **Performance**
  - Depth estimation may take 5-30 seconds depending on image size and hardware
  - Stereogram generation should be relatively fast (< 5 seconds for typical images)
  - Support for GPU acceleration when available (CUDA for NVIDIA)
  - Automatic device selection (CPU fallback if GPU unavailable)
  - Model download size: ~100-500MB depending on variant (consider disk space)

- **Dependencies**
  - PyTorch (for DepthAnything model)
  - torchvision (may be needed for image preprocessing)
  - Pillow/PIL (for image I/O)
  - NumPy (for array operations)
  - transformers (HuggingFace, for DepthAnything model loading)
  - huggingface_hub (for model downloading, included with transformers)
  - tqdm (for progress bars, included with transformers)
  - Standard library (argparse, pathlib, etc.)

- **Compatibility**
  - Cross-platform (macOS, Linux, Windows)
  - Handle different image formats gracefully
  - Work with both color and grayscale depth maps
  - Support both x86_64 and ARM64 architectures

### Edge Cases & Error Handling

- **Invalid Input**
  - Non-existent input file path → clear error message
  - Invalid image format → error message with supported formats
  - Corrupted image file → graceful error handling

- **Model Loading & Downloading**
  - First-time model download → show progress indicator, handle timeout
  - Model download interruption → resume from checkpoint if supported, or clear error
  - Insufficient disk space for model → check before download, clear error message
  - Network issues during model download → retry with exponential backoff
  - Model cache corruption → detect and re-download if needed
  - Model version mismatch → handle gracefully or warn user
  - Offline mode → use cached model if available, clear error if not

- **Image Processing**
  - Very large images → automatically resize to exactly 1 megapixel (1MP) before processing to manage memory constraints
  - Very small images → ensure minimum viable size of 1MP
  - Unusual aspect ratios → maintain aspect ratio when resizing

- **Output**
  - Invalid output path → check directory exists, create if needed
  - Insufficient disk space → error before processing
  - Permission errors → clear error message

- **Depth Map Issues**
  - Uniform depth (no variation) → handle gracefully, still generate stereogram
  - Extreme depth values → normalize appropriately

## Technical Approach

### Implementation Strategy

**Architecture:**
- Modular design with clear separation of concerns
- Three main components:
  1. Depth estimation module
  2. Autostereogram generation module
  3. CLI interface module

**Project Structure:**
```
stereogram-generator/
├── stereogram_generator/
│   ├── __init__.py
│   ├── depth_estimator.py      # DepthAnything integration
│   ├── stereogram.py            # Noise-based autostereogram generation
│   └── cli.py                   # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_depth_estimator.py
│   ├── test_stereogram.py
│   └── test_cli.py
├── docs/
│   ├── spec.md                  # This specification
│   └── research_findings.md     # Detailed research findings and technical details
├── venv/                        # Virtual environment (gitignored)
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py                     # Package setup for installable CLI command
```

**Technology Choices:**
- **DepthAnything**: Use HuggingFace transformers for model loading
  - Model: `depth-anything/Depth-Anything-V2-Base-hf` (V2-base variant, 97.5M parameters)
  - Model architecture: DPT (Dense Prediction Transformer) with DINOv2 backbone
  - Model download: Use HuggingFace `transformers` library (uses `huggingface_hub` internally with automatic retry and progress indication)
  - Output format: `predicted_depth` tensor requiring interpolation to original image size
- **PyTorch**: Support CUDA for NVIDIA GPUs on Linux/Windows
  - Fallback to CPU if CUDA unavailable
- **Image Processing**: Pillow (PIL) for I/O, NumPy for array operations (vectorization where possible for performance)
- **CLI**: argparse from standard library
- **Testing**: pytest (following project preference)
- **Code Quality**: Consider black/flake8 for formatting (optional)
- **Model Download**: `transformers` library uses `tqdm` internally for progress indication (may need custom implementation if progress not visible)

**Autostereogram Algorithm:**
1. Generate initial random grayscale noise pattern (single column or narrow strip, width configurable via `--noise-width`, default: 32-64 pixels based on image width)
2. Normalize depth map to [0, 1] range using min-max normalization
3. Invert depth map if needed (hardcoded based on DepthAnything V2-base output characteristics - typically brighter = closer)
4. Scale depth map to appropriate shift range (configurable via `--shift-range`, default: 5-20 pixels based on image width)
5. For each row:
   - Seed the first columns (width = noise pattern width) with the noise pattern, tiling vertically if needed
   - For each subsequent pixel position:
     - Calculate horizontal shift: `shift = int(depth_value * shift_amplitude * pattern_width)`
     - Copy pixel from position `(current_column - pattern_width + shift)` with wrapping
     - This ensures continuity by referencing previously placed pixels
6. The shift amount determines perceived depth: larger shifts = closer objects (objects appear to pop out)

**Depth Map Processing:**
- Load and preprocess input image for DepthAnything
- Resize image to exactly 1 megapixel (1MP), maintaining aspect ratio using bicubic interpolation (resize before depth estimation for efficiency)
- Use `AutoImageProcessor.from_pretrained()` to normalize image for model input (handles normalization automatically)
- Run inference to get depth map using DepthAnything V2-base model
- Output is a `predicted_depth` tensor that requires interpolation to original image size using `torch.nn.functional.interpolate` with `mode="bicubic"`
- Convert tensor to NumPy array for processing
- Normalize depth values to [0, 1] range using min-max normalization
- Invert depth map if needed (hardcoded based on DepthAnything V2-base output characteristics - typically brighter values = closer objects, which aligns with autostereogram requirements, but may need inversion depending on actual model output)
- Scale normalized depth map to appropriate shift range for stereogram generation (using `--shift-range` parameter)

### Affected Components

**New Components (to be created):**
- `stereogram_generator/depth_estimator.py`: DepthAnything model loading and inference
- `stereogram_generator/stereogram.py`: Noise-based autostereogram generation algorithm
- `stereogram_generator/cli.py`: Command-line argument parsing and orchestration
- `stereogram_generator/__init__.py`: Package initialization

**Configuration Files:**
- `requirements.txt`: Python dependencies
- `README.md`: Installation and usage instructions
- `.gitignore`: Exclude venv, __pycache__, model cache, etc.
- `setup.py`: Package setup for installable CLI command (entry point: `mystereogram` command)

**Test Files:**
- `tests/test_depth_estimator.py`: Test depth estimation with sample images
- `tests/test_stereogram.py`: Test stereogram generation with synthetic depth maps
- `tests/test_cli.py`: Test CLI argument parsing and error handling
- Note: Detailed test case specifications are not included in this specification document; test implementation will determine specific test cases during development

### Dependencies & Integration

**External Dependencies:**
- `torch` (PyTorch): For DepthAnything model inference
  - Must support MPS backend for Apple Silicon (PyTorch 1.12+)
  - CUDA support optional but recommended for NVIDIA GPUs
- `torchvision`: May be needed for image preprocessing
- `transformers` (HuggingFace): If using HuggingFace model loading
- `huggingface_hub`: For reliable model downloading with progress tracking
- `Pillow`: Image I/O and basic processing
- `numpy`: Array operations for depth maps and stereogram generation
- `tqdm`: Progress bars for model download and processing (optional but recommended)
- `pytest`: Testing framework (dev dependency)

**Model Dependencies:**
- DepthAnything model weights (downloaded automatically on first use)
- Model cache location: 
  - HuggingFace: `~/.cache/huggingface/hub/` (configurable via `HF_HOME` env var)
  - PyTorch Hub: `~/.cache/torch/hub/` (if using PyTorch Hub)
- Model size: ~97.5M parameters (F32 format), approximately 390MB on disk for V2-base variant
- Download progress: Show progress bar during download
- Cache management: Verify cached model integrity, handle corrupted cache

**Device Support:**
- **CPU**: Universal fallback, works everywhere
- **CUDA**: NVIDIA GPUs on Linux/Windows (if PyTorch with CUDA installed)
- Automatic device detection and selection

**No External Services:**
- All processing is local
- Model download happens once, then cached locally
- No internet required after initial model download

## Acceptance Criteria

- [ ] CLI accepts input image path and output path as arguments
- [ ] Application successfully loads DepthAnything model (with download on first run)
- [ ] Depth map is correctly computed from input image
- [ ] Noise-based autostereogram is generated from depth map
- [ ] Output image is saved to specified path in valid image format
- [ ] Error handling works for invalid input files, missing paths, etc.
- [ ] Help text is displayed with `--help` flag
- [ ] Unit tests pass for depth estimation module
- [ ] Unit tests pass for stereogram generation module
- [ ] Unit tests pass for CLI argument parsing
- [ ] README provides clear installation and usage instructions
- [ ] Application works on macOS, Linux, and Windows
- [ ] Model downloads successfully on first run with progress indication
- [ ] Model caching works correctly (subsequent runs use cached model)
- [ ] CPU fallback works when GPU unavailable
- [ ] Images are automatically resized to 1MP while maintaining aspect ratio
- [ ] CLI command runs directly as `mystereogram` (without `python -m`)

## Implementation Tasks

### Development Order & Dependencies

The implementation is organized into 8 phases in strict linear order for single-developer workflow:

```
Phase 1 (Foundation)
  └─> Phase 2 (Utilities)
       └─> Phase 3 (Model Loading)
            └─> Phase 4 (Inference)
                 └─> Phase 5 (Stereogram Generation)
                      └─> Phase 6 (Integration Testing)
                           └─> Phase 7 (CLI Interface)
                                └─> Phase 8 (Documentation)
```

**Linear Development Flow:**
- Each phase must be completed before moving to the next
- Phases build incrementally on previous work
- Unit tests are written as each module is completed
- Integration testing validates the full pipeline before CLI development
- Documentation is finalized after the CLI is working

### Phase 1: Foundation & Setup
**Dependencies: None**

- [ ] Create project directory structure (`stereogram_generator/`, `tests/`, `docs/`)
- [ ] Initialize git repository (if not exists)
- [ ] Create `.gitignore` (venv, __pycache__, model cache, `.pytest_cache/`, `*.pyc`, etc.)
- [ ] Create `requirements.txt` with dependencies (torch, transformers, pillow, numpy, pytest)
- [ ] Create basic `README.md` structure
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Verify PyTorch installation supports CUDA (if applicable)
- [ ] Create package `__init__.py` files (`stereogram_generator/__init__.py`, `tests/__init__.py`)
- [ ] Generate sample test images using ImageMagick (`magick` command) for workflow validation:
  - Create `tests/fixtures/` directory for test images
  - Generate simple black and white primitives (circles, squares, gradients)
  - Examples: circular depth patterns, rectangular shapes, linear gradients
  - These simple images will help validate depth estimation and stereogram generation early

### Phase 2: Core Utilities & Device Support
**Dependencies: Phase 1 (dependencies installed)**

- [ ] Create `stereogram_generator/utils.py` for shared utilities
- [ ] Implement device detection utility (CPU/CUDA) with automatic selection
  - Use `torch.cuda.is_available()`
  - Return device string for use by other modules
- [ ] Implement image preprocessing utilities:
  - Resize to exactly 1MP with bicubic interpolation (maintain aspect ratio)
  - Validate image formats and handle errors
- [ ] Write unit tests for device detection
- [ ] Write unit tests for image preprocessing (resize, format validation)

### Phase 3: Depth Estimation Module (Part 1 - Model Loading)
**Dependencies: Phase 2 (device detection, image preprocessing)**

- [ ] Create `stereogram_generator/depth_estimator.py` skeleton
- [ ] Implement model and processor initialization:
  - `AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")`
  - `AutoImageProcessor.from_pretrained(...)`
- [ ] Implement model loading with device support (move model to selected device)
- [ ] Test model download (first run) with progress indication
- [ ] Test model caching (subsequent runs use cached model)
- [ ] Add error handling for model loading and download failures
- [ ] Write unit tests for model loading (with mocked download)

### Phase 4: Depth Estimation Module (Part 2 - Inference & Processing)
**Dependencies: Phase 3 (model loaded successfully)**

- [ ] Implement depth map inference:
  - Preprocess image (resize to 1MP, normalize via AutoImageProcessor)
  - Run inference with device support
  - Get `predicted_depth` tensor from model output
- [ ] Implement depth map post-processing:
  - Interpolate to original image size using `torch.nn.functional.interpolate` with `mode="bicubic"`
  - Convert tensor to NumPy array
  - Normalize to [0, 1] range using min-max normalization
- [ ] Test depth map inversion requirement:
  - Run inference on sample images
  - Determine if DepthAnything V2-base outputs brighter=closer or darker=closer
  - Implement inversion if needed (hardcoded based on findings)
- [ ] Add error handling for inference failures
- [ ] Write unit tests for inference and post-processing (with mocked model)

### Phase 5: Autostereogram Generation Module
**Dependencies: Phase 4 (depth map processing complete)**

- [ ] Create `stereogram_generator/stereogram.py`
- [ ] Implement random grayscale noise pattern generation:
  - Generate pattern with configurable width (default: 32-64 pixels based on image size)
  - Pattern should be tiled vertically for rows
- [ ] Implement core autostereogram algorithm:
  - Accept normalized depth map [0, 1] and noise pattern
  - For each row:
    - Seed first columns (width = pattern width) with noise pattern
    - For subsequent pixels: calculate shift and copy from `(current_column - pattern_width + shift)`
  - Implement depth-to-shift mapping: `shift = int(depth_value * shift_amplitude * pattern_width)`
  - Handle wrapping at image edges
- [ ] Implement parameter calculation utilities:
  - Auto-calculate noise width: `max(64, min(128, image_width // 15))`
  - Auto-calculate shift range: `max(20, min(60, image_width // 30))`
- [ ] Handle edge cases:
  - Uniform depth (no variation)
  - Extreme depth values
  - Edge wrapping
- [ ] Optimize for performance (vectorization where possible with NumPy)
- [ ] Write unit tests with synthetic depth maps:
  - Circular patterns
  - Gradients
  - Uniform depth
  - Edge cases
- [ ] Test with real depth maps from Phase 4 to validate integration

### Phase 6: Integration & End-to-End Testing
**Dependencies: Phase 5 (stereogram generation complete)**

- [ ] Create integration test script to validate full pipeline:
  - Load real image
  - Generate depth map
  - Generate stereogram
  - Save output
- [ ] Test end-to-end workflow with sample images
- [ ] Verify output quality and 3D effect visibility
- [ ] Test with various image sizes (small, medium, large)
- [ ] Test with different image formats (JPEG, PNG)
- [ ] Test error handling paths (invalid input, missing files, etc.)
- [ ] Benchmark performance on different devices (CPU, CUDA if available)
- [ ] Test offline mode (cached model without network)

### Phase 7: CLI Interface
**Dependencies: Phase 6 (validated pipeline)**

- [ ] Create `stereogram_generator/cli.py` with argparse setup
- [ ] Implement argument parsing:
  - Required: `INPUT`, `OUTPUT`
  - Optional: `--noise-width`, `--shift-range`, `--device`, `--save-depth-map`, `--clear-cache`, `--help`
- [ ] Implement main orchestration logic:
  - Load and validate input image
  - Generate depth map
  - Generate stereogram
  - Save output
  - Handle optional depth map saving
- [ ] Add progress messages and error reporting
- [ ] Configure `setup.py` entry point for direct `mystereogram` command execution
- [ ] Test CLI installation (`pip install -e .`)
- [ ] Write CLI tests (argument parsing, error handling)
- [ ] Test CLI with real images

### Phase 8: Documentation & Polish
**Dependencies: Phase 7 (working CLI)**

- [ ] Complete README with:
  - Installation instructions
  - Usage examples
  - Parameter tuning guide
  - Troubleshooting section
- [ ] Document algorithm parameters and tuning recommendations
- [ ] Add code comments for complex logic
- [ ] Add docstrings to all public functions and classes
- [ ] Create example usage commands in README
- [ ] Verify cross-platform compatibility (macOS, Linux, Windows)
- [ ] Add example images or links to test images
- [ ] Final code review and cleanup

## Risk Assessment

### Potential Issues

**Model Loading & Compatibility**
- **Risk**: DepthAnything model format or API may change, or may not be easily accessible
- **Impact**: High - core functionality depends on this
- **Mitigation**: Research multiple approaches (HuggingFace, PyTorch Hub, direct model files), have fallback options

**Model Download & Caching**
- **Risk**: Large model downloads may fail, timeout, or be interrupted
- **Impact**: High - prevents first-time use
- **Mitigation**: Implement retry logic, progress indication, resume capability if supported, clear error messages


**Performance & Memory**
- **Risk**: Large images may cause memory issues or very slow processing
- **Impact**: Medium - affects usability
- **Mitigation**: Automatically resize images to exactly 1 megapixel (1MP) before processing, add progress indicators

**Stereogram Quality**
- **Risk**: Generated autostereograms may not produce visible 3D effect
- **Impact**: Medium - affects core value proposition
- **Mitigation**: Research proven algorithms, test with various depth maps, allow parameter tuning

**Dependency Conflicts**
- **Risk**: PyTorch/transformers versions may conflict or have compatibility issues
- **Impact**: Medium - affects installation
- **Mitigation**: Pin specific versions in requirements.txt, test on clean environment


**Cross-Platform Issues**
- **Risk**: Path handling or model cache locations may differ across platforms
- **Impact**: Low - affects usability but not core functionality
- **Mitigation**: Use pathlib for paths, test on multiple platforms

### Mitigation Strategies

- **Prototype First**: Create minimal working prototype to validate DepthAnything integration before full implementation
- **Incremental Development**: Build and test each module independently
- **Version Pinning**: Pin dependency versions in requirements.txt to avoid breaking changes
- **Error Messages**: Provide clear, actionable error messages for common failure modes
- **Documentation**: Document known issues and workarounds in README

### Investigation Requirements

**Before Implementation:**
- [x] Verify DepthAnything V2-base model availability on HuggingFace (`depth-anything/Depth-Anything-V2-Base-hf`) - **COMPLETED**: Model confirmed available, 97.5M parameters
- [x] Test model download process and verify HuggingFace hub integration - **COMPLETED**: `transformers` library handles download with built-in retry and progress indication
- [x] Research noise-based autostereogram algorithms and validate approach - **COMPLETED**: Algorithm validated, implementation details documented
- [ ] Test DepthAnything V2-base on sample images to understand output format and normalization needs - **PARTIAL**: Output format known (predicted_depth tensor), actual inversion requirement to be tested
- [x] Research optimal noise pattern width ratio to image size for best stereogram quality - **COMPLETED**: Default `max(64, min(128, image_width // 15))` implemented
- [x] Research optimal shift range ratio to image size for depth-to-stereogram conversion - **COMPLETED**: Default `max(20, min(60, image_width // 30))` implemented
- [x] Research if existing Python libraries can be used (e.g., pystereogram) or if custom implementation needed - **COMPLETED**: Custom implementation recommended for better control
- [x] Verify model download size and cache location behavior for V2-base variant - **COMPLETED**: ~390MB, cached at `~/.cache/huggingface/hub/`
- [x] Test model download retry and error handling scenarios - **COMPLETED**: Built into `huggingface_hub` with exponential backoff

**Implementation Notes:**

The phases follow a strict linear sequence optimized for single-developer workflow:

1. **Phases 1-2**: Foundation and utilities - establish base infrastructure
2. **Phase 3**: Model loading - validate model download and caching before building inference
3. **Phase 4**: Depth estimation inference - complete depth map generation pipeline
4. **Phase 5**: Stereogram generation - build on real depth maps from Phase 4
5. **Phase 6**: Integration testing - validate full pipeline end-to-end before CLI
6. **Phase 7**: CLI interface - build user-facing interface on validated pipeline
7. **Phase 8**: Documentation - final polish and documentation

**Key Validation Points:**
- After Phase 3: Verify model loads successfully and caches properly
- After Phase 4: Test depth map output format, determine inversion requirement, validate with real images
- After Phase 5: Test stereogram generation with real depth maps, verify 3D effect quality
- After Phase 6: Validate full pipeline with various images, test error handling

## Notes

- This is a relatively simple CLI application, so architecture should remain straightforward
- Focus on correctness and usability over advanced features
- The noise-based approach is simpler than pattern-image-based and matches user requirements
- Model download is a critical first-run experience - make it smooth with clear progress indication
- Model caching should be transparent to users but robust (handle corruption, verify integrity)
- Future enhancements could include: batch processing, different stereogram styles, GUI interface

## Research & References

Detailed research findings, algorithm implementation details, and technical references are documented in `docs/research_findings.md`. Key findings include:
- Correct model path: `depth-anything/Depth-Anything-V2-Base-hf` (97.5M parameters)
- Optimal default parameters for noise-width and shift-range based on image size
- Autostereogram algorithm implementation details with code examples
- Depth map processing pipeline and normalization requirements
- HuggingFace integration details and model download behavior

