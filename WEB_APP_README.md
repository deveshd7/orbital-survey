# 🌐 Orbital Survey - Interactive Web App

D3.js-powered interactive version of Orbital Survey V4!

## Features

### 🎨 Interactive Visualizations
- **Animated Color Bars** - Hover to see details, click to highlight
- **Interactive Scatter Plot** - Zoom, pan, hover for color info
- **Live Data** - Real-time analysis updates
- **Smooth Animations** - D3.js transitions and effects

### 📊 Analysis Panels
1. **Source Image** - Preview with stats (resolution, aspect ratio)
2. **Chromatic Distribution** - K-means color extraction with percentages
3. **Hue × Saturation Map** - Interactive scatter plot showing color space
4. **Analysis Summary** - Contours, vectors, features, gradients

### 💾 Export
- Export full V4 static image (PNG)
- Download analysis data (JSON)

## Quick Start

### 1. Install Dependencies

```bash
# Base requirements
pip install -r requirements.txt

# Web app requirements
pip install -r requirements_web.txt
```

Or install all at once:
```bash
pip install flask werkzeug numpy opencv-python Pillow scipy scikit-learn
```

### 2. Run the Web App

```bash
python web_app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

1. **Upload Image**
   - Drag & drop an image
   - Or click to browse
   - Supports: JPG, PNG, WebP, BMP (max 16MB)

2. **View Analysis**
   - See interactive color distribution
   - Explore hue/saturation scatter plot
   - Hover over elements for details

3. **Export**
   - Click "Export Full Analysis" for complete PNG
   - Or analyze another image

## Architecture

```
┌─────────────┐         ┌──────────────┐
│   Browser   │◄───────►│ Flask Server │
│   (D3.js)   │  JSON   │   (Python)   │
└─────────────┘         └──────────────┘
      │                        │
      │                        ▼
      │                 ┌──────────────┐
      │                 │ orbital_     │
      │                 │ survey_v4.py │
      │                 └──────────────┘
      │                        │
      ▼                        ▼
┌─────────────┐         ┌──────────────┐
│ Interactive │         │ OpenCV       │
│ Charts      │         │ NumPy        │
│ Animations  │         │ scikit-learn │
└─────────────┘         └──────────────┘
```

## API Endpoints

### POST /api/analyze
Analyzes uploaded image and returns JSON data.

**Request:**
- `image`: Image file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "image": {
    "width": 1920,
    "height": 1080,
    "base64": "..."
  },
  "colors": [
    {
      "rgb": [255, 128, 64],
      "hex": "#FF8040",
      "percentage": 23.5,
      "hue": 30.5,
      "saturation": 74.9,
      "value": 100.0
    }
  ],
  "scatter": [
    {
      "hue": 45.2,
      "saturation": 67.8,
      "value": 89.3,
      "rgb": [228, 195, 73]
    }
  ],
  "stats": {
    "contour_count": 342,
    "vector_count": 1245,
    "avg_gradient": 23.7,
    "feature_count": 180
  }
}
```

### POST /api/export
Generates full V4 PNG image.

**Request:**
- `image`: Image file (multipart/form-data)

**Response:**
- PNG image file (download)

## D3.js Visualizations

### Color Distribution Chart
- **Type:** Horizontal bar chart
- **Features:**
  - Animated entry (staggered)
  - Hover effects
  - Percentage labels
  - Hex codes
- **Interaction:** Hover to highlight

### Hue-Saturation Scatter
- **Type:** 2D scatter plot
- **Features:**
  - 5000+ sample points
  - Color-coded dots
  - Dominant colors marked
  - Hue rainbow gradient
  - Grid overlay
  - Tooltips on hover
- **Interaction:**
  - Hover for RGB/HSV details
  - Click dominant colors for info

## Customization

### Adjust Sample Count
In `web_app.py`, line 92:
```python
hsv_data = osv4.compute_hs_scatter(img_resized, sample_count=5000)
```
- Increase for more detail (slower)
- Decrease for faster analysis

### Change Color Clusters
In `web_app.py`, modify V4 config:
```python
config.NUM_COLORS = 10  # Change to 5, 8, 12, etc.
```

### Modify Visualization Colors
In `static/css/style.css`:
```css
/* Primary color */
--color-primary: #00b4a0;

/* Gradient background */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

## Performance

- **Small images (< 1MB):** ~1-2 seconds
- **Medium images (1-5MB):** ~2-4 seconds
- **Large images (5-15MB):** ~4-8 seconds

### Optimization Tips
1. Resize large images before upload
2. Use JPG instead of PNG for photos
3. Close other browser tabs
4. Use Chrome/Edge for best D3.js performance

## Deployment

### Local Network
```bash
python web_app.py
# Access from other devices: http://YOUR_IP:5000
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r requirements_web.txt
EXPOSE 5000
CMD ["python", "web_app.py"]
```

## Browser Compatibility

- ✅ Chrome 90+ (Recommended)
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+
- ❌ IE 11 (not supported)

## Troubleshooting

### Port Already in Use
```bash
# Change port in web_app.py, line ~115
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Upload Fails
- Check file size (< 16MB)
- Check file type (JPG, PNG, WebP, BMP only)
- Check server logs for errors

### Slow Performance
- Reduce sample count in `compute_hs_scatter()`
- Use smaller images
- Disable debug mode in production

### D3.js Not Loading
- Check internet connection (D3.js loaded from CDN)
- Or download D3.js locally:
  ```html
  <script src="/static/js/d3.v7.min.js"></script>
  ```

## Comparison: Web vs CLI

| Feature | Web App | V4 CLI |
|---------|---------|--------|
| Interactive charts | ✅ | ❌ |
| Animations | ✅ | ❌ |
| Hover details | ✅ | ❌ |
| Export options | JSON + PNG | PNG only |
| Setup time | 2 min | 30 sec |
| Performance | Slower | Faster |
| Best for | Exploration | Batch processing |

## Future Enhancements

- [ ] Side-by-side image comparison
- [ ] Adjustable parameters (live sliders)
- [ ] Batch upload (analyze multiple images)
- [ ] Color palette export (Sketch, Figma)
- [ ] SVG export for vectors
- [ ] WebGL acceleration for large datasets
- [ ] Real-time webcam analysis

## Credits

- **Backend:** Python + Flask
- **Analysis:** OpenCV, NumPy, scikit-learn
- **Visualization:** D3.js v7
- **UI:** Custom CSS with gradients
