# Orbital Survey Enhancement Plan

## Critical Fixes

### 1. Resolution Preservation
**Current Issue:** Images downsample from 1920px to ~1248px
**Fix:**
- Add `SCALE_FACTOR` config (1.0 = preserve input size, 2.0 = double, etc.)
- Calculate output size based on input dimensions
- Add `MIN_OUTPUT_WIDTH` and `MAX_OUTPUT_WIDTH` constraints
- Option to upscale small images

### 2. Dynamic Layout System
**Current Issue:** Fixed 65/35% split fails for portraits and wide panoramas
**Fix:**
- Detect orientation (portrait/landscape/square)
- Use different layouts per orientation:
  - **Landscape:** Side panel (current style)
  - **Portrait:** Bottom panel or floating panels
  - **Wide panorama:** Top/bottom panels
- Calculate panel size based on content needs, not fixed percentages

### 3. Adaptive Canvas Sizing
**Current Issue:** Canvas height can cut off tall images
**Fix:**
- Calculate canvas from image + panel + padding requirements
- Ensure all panels fit without overflow
- Add min/max canvas size constraints

## Quality Improvements

### 4. Better Image Processing
- **Preserve aspect ratio strictly** (currently can distort slightly)
- **Add sharpening** after resize to combat blur
- **Adjustable overlay opacity** in config
- **Optional background blur** instead of white blend

### 5. Smarter Feature Detection
- **Adaptive thresholds** based on image content
- **Region-of-interest** weighting (detect important areas)
- **Clustering** to avoid feature clumping
- **Edge preservation** for better contours

### 6. Performance Optimizations
- **Multi-threading** for layer generation (parallel processing)
- **Caching** FFT and gradient computations
- **Progress callbacks** for GUI integration
- **Batch processing** mode

### 7. Better CLI/UX
- **argparse** instead of sys.argv
- **Rich progress bars** (using `rich` or `tqdm`)
- **Preset configs** (--style sci-fi, --style minimal, etc.)
- **Preview mode** (low-res fast preview)
- **Config file support** (YAML/JSON)

### 8. Output Options
- **Layered PSD export** (separate layers for editing)
- **Animation frames** (for video processing)
- **Metadata embedding** (processing parameters in EXIF)
- **Before/after comparison** output

## Code Quality

### 9. Error Handling
- Validate input file exists and is readable
- Check for supported image formats
- Handle corrupted images gracefully
- Memory checks for large images
- Informative error messages

### 10. Testing & Documentation
- Unit tests for each layer function
- Example gallery with different image types
- Performance benchmarks
- API documentation for library use

## Advanced Features (Optional)

### 11. AI Enhancements
- **Semantic segmentation** for region-based analysis
- **Object detection** for focused feature extraction
- **Style transfer** integration
- **Depth estimation** for 3D-style overlays

### 12. Interactive Features
- **Web UI** (Flask/Streamlit)
- **Real-time preview** with slider controls
- **Parameter tweaking** interface
- **Comparison slider** (before/after)

### 13. Export Variations
- **Print-ready** high-DPI output (300 DPI)
- **Social media** presets (Instagram square, Twitter banner, etc.)
- **Video frame sequence** generation
- **Transparent overlay** export (for compositing)

## Implementation Priority

**Phase 1: Critical Fixes (Do First)**
1. ✅ Resolution preservation system
2. ✅ Dynamic layout for all orientations
3. ✅ Proper canvas sizing
4. ✅ Error handling

**Phase 2: Quality (Do Next)**
5. Adjustable overlay parameters
6. Better CLI with argparse
7. Progress indicators
8. Config file support

**Phase 3: Advanced (If Needed)**
9. Multi-threading
10. Batch processing
11. Web UI
12. Additional export formats

## Should You Stick with Python?

**YES! Python is ideal for this because:**
- ✅ Best libraries for computer vision (OpenCV, scikit-image)
- ✅ Rapid prototyping and iteration
- ✅ Excellent for creative/research tools
- ✅ Easy to add ML/AI features later
- ✅ Great for GUIs (tkinter, PyQt, Streamlit)
- ✅ Cross-platform with no changes

**When to consider alternatives:**
- ❌ Real-time processing (use C++/Rust with Python bindings)
- ❌ Mobile apps (use native or Flutter)
- ❌ Embedded systems (use C/Rust)

For your use case (creative image analysis tool), **Python is perfect**.

## Recommended Stack Additions

- **Rich/tqdm**: Beautiful progress bars
- **click/argparse**: Better CLI
- **pydantic**: Config validation
- **streamlit**: Quick web UI (if needed)
- **pytest**: Testing framework
- **black/ruff**: Code formatting
