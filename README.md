# Orbital Survey

Transform photographs into sci-fi planetary survey visualizations with technical HUD overlays.

![Example](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- 🎨 **6 Analysis Layers**: Contours, gradients, FFT spectrum, color extraction, feature detection, HUD overlay
- 📐 **Adaptive Layouts**: Auto-detects portrait/landscape/panorama and adjusts panel placement
- 🔍 **Resolution Preservation**: Maintains full input resolution (no downsampling)
- ⚙️ **Highly Configurable**: Adjust overlay intensity, detail levels, colors
- 🖼️ **Professional Output**: High-quality PNG/JPEG with white background theme

## Examples

**Input:** Regular photograph
**Output:** Sci-fi technical analysis visualization with:
- Luminance contours (topographic-style lines)
- Gradient vector field (directional arrows)
- 2D FFT frequency spectrum
- Dominant color extraction
- Feature constellation mesh (Harris corners + Delaunay triangulation)
- Coordinate grid and HUD elements

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/orbital-survey.git
cd orbital-survey

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Interactive mode (file dialog)
python orbital_survey_enhanced.py

# Command line
python orbital_survey_enhanced.py input.jpg output.png

# Auto-naming
python orbital_survey_enhanced.py input.jpg
```

## Configuration

Edit the `Config` class in `orbital_survey_enhanced.py`:

```python
class Config:
    # Resolution
    SCALE_FACTOR = 1.0          # 1.0 = preserve, 2.0 = double
    MAX_OUTPUT_WIDTH = 3840     # 4K max

    # Overlays
    BASE_IMAGE_FADE = 0.15      # Image fade amount
    OVERLAY_ALPHA_BOOST = 1.0   # Overlay brightness

    # Detail
    CONTOUR_LEVELS = 20         # Number of contour lines
    MAX_FEATURES = 200          # Feature points
    VECTOR_GRID_SPACING = 30    # Vector density
```

## Version Comparison

### Enhanced Version (Recommended)
- ✅ Full resolution preservation
- ✅ Adaptive layouts (portrait/landscape/panorama)
- ✅ Better error handling
- ✅ Configurable scaling

### Original Version
- Fixed output width (1920px)
- Side panel only
- Basic configuration

## Requirements

- Python 3.8+
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- Pillow >= 10.0.0
- scipy >= 1.11.0
- scikit-learn >= 1.3.0

## Documentation

- [`QUICK_START.md`](QUICK_START.md) - Setup and usage guide
- [`README_ENHANCED.md`](README_ENHANCED.md) - Feature comparison and details
- [`ENHANCEMENT_PLAN.md`](ENHANCEMENT_PLAN.md) - Future improvements roadmap
- [`SUMMARY.md`](SUMMARY.md) - Complete analysis summary

## How It Works

1. **Contours**: Gaussian smoothing + marching squares for topographic lines
2. **Gradients**: Sobel operators for directional intensity changes
3. **FFT**: 2D Fast Fourier Transform with log scaling
4. **Colors**: K-means clustering for dominant color extraction
5. **Features**: Harris corner detection + Delaunay triangulation
6. **HUD**: Coordinate grids, brackets, technical readouts

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! See issues for planned improvements.

## Credits

Created with Python, OpenCV, NumPy, SciPy, and scikit-learn.
