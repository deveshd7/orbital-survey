# Quick Start Guide

## Setup (First Time)

### 1. Create Fresh Virtual Environment
```bash
# Remove old venv
rm -rf venv

# Create new venv
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - numerical operations
- `opencv-python` - image processing
- `Pillow` - image I/O
- `scipy` - scientific computing
- `scikit-learn` - machine learning (K-means)

## Usage

### Run Original Version
```bash
source venv/bin/activate
python orbital_survey.py test.png
```

### Run Enhanced Version (Recommended)
```bash
source venv/bin/activate
python orbital_survey_enhanced.py test.png
```

**Key improvements in enhanced version:**
- ✅ Full resolution preservation
- ✅ Adaptive layouts (portrait/landscape/panorama)
- ✅ Better error handling
- ✅ More configurable

## Quick Test

```bash
# Test on panorama image (1920×862)
python orbital_survey_enhanced.py test11.png

# Test on portrait image (1920×2378)
python orbital_survey_enhanced.py test.png
```

## Compare Outputs

**Original:**
- Downsamples to ~1248px width
- Fixed side panel layout
- May have layout issues

**Enhanced:**
- Preserves full 1920px resolution
- Smart layout selection
- Proper aspect ratio handling

## Configuration

Edit `Config` class in `orbital_survey_enhanced.py`:

```python
# Resolution
SCALE_FACTOR = 1.0      # Increase for upscaling (2.0 = 2x)
MAX_OUTPUT_WIDTH = 3840 # 4K maximum

# Overlays
BASE_IMAGE_FADE = 0.15      # Image fade (0=none, 1=white)
OVERLAY_ALPHA_BOOST = 1.0   # Overlay brightness

# Detail level
CONTOUR_LEVELS = 20         # More = denser contours
MAX_FEATURES = 200          # More = denser constellation
VECTOR_GRID_SPACING = 30    # Lower = denser vectors
```

## Troubleshooting

### Import Errors
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### Low Resolution Output
```python
# In Config class, set:
SCALE_FACTOR = 1.0  # Or higher
MAX_OUTPUT_WIDTH = 3840  # Increase if needed
```

### Wrong Layout
```python
# Check auto-detection thresholds:
LANDSCAPE_THRESHOLD = 1.3
PORTRAIT_THRESHOLD = 0.75
PANORAMA_THRESHOLD = 2.5
```

### Overlays Too Faint
```python
OVERLAY_ALPHA_BOOST = 1.5
BASE_IMAGE_FADE = 0.25
```

## Next Steps

1. ✅ Test enhanced version on your images
2. ✅ Adjust Config values to your preference
3. See `ENHANCEMENT_PLAN.md` for future improvements
4. See `README_ENHANCED.md` for detailed comparison
