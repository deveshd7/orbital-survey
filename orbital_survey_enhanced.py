"""
ORBITAL SURVEY PROTOCOL - ENHANCED VERSION
===========================================
Transforms landscape photographs into sci-fi planetary survey visualizations.

ENHANCEMENTS:
- Resolution preservation (no downsampling)
- Dynamic layouts for portrait/landscape/panorama
- Adaptive panel sizing
- Better configurability
- Error handling and validation
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from pathlib import Path
import colorsys
import sys
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class LayoutMode(Enum):
    """Layout orientation modes"""
    LANDSCAPE = "landscape"  # Side panel (original style)
    PORTRAIT = "portrait"    # Bottom panel
    PANORAMA = "panorama"    # Top panel for ultra-wide


class Config:
    # Resolution Control
    SCALE_FACTOR = 1.0          # 1.0 = preserve input size, 0.5 = half, 2.0 = double
    MAX_OUTPUT_WIDTH = 3840     # 4K max width
    MIN_OUTPUT_WIDTH = 1280     # Minimum output
    PRESERVE_ASPECT = True      # Strict aspect ratio preservation

    # Auto-detect layout based on aspect ratio
    AUTO_LAYOUT = True
    LANDSCAPE_THRESHOLD = 1.3   # width/height > this = landscape
    PORTRAIT_THRESHOLD = 0.75   # width/height < this = portrait
    PANORAMA_THRESHOLD = 2.5    # width/height > this = panorama

    # Panel sizing (percentages)
    PANEL_SIZE_LANDSCAPE = 0.30  # 30% of width for side panel
    PANEL_SIZE_PORTRAIT = 0.25   # 25% of height for bottom panel
    PANEL_SIZE_PANORAMA = 0.20   # 20% of height for top panel

    # Padding
    PADDING = 80
    PANEL_PADDING = 40

    # Colors (RGBA) - white background theme
    PRIMARY = (0, 180, 160, 255)       # Teal
    SECONDARY = (220, 80, 40, 255)     # Red-orange
    TERTIARY = (120, 60, 200, 255)     # Purple
    DIM = (100, 100, 100, 200)         # Gray for text
    BACKGROUND = (255, 255, 255, 255)  # White

    # Overlay intensity
    BASE_IMAGE_FADE = 0.15      # 0.0 = original, 1.0 = white
    OVERLAY_ALPHA_BOOST = 1.0   # Multiplier for overlay opacity

    # Contours
    CONTOUR_LEVELS = 20
    CONTOUR_SMOOTHING = 3.0
    CONTOUR_LINE_WIDTH = 2

    # Vector field
    VECTOR_GRID_SPACING = 30
    VECTOR_SCALE = 20
    VECTOR_MIN_THRESHOLD = 0.03

    # Feature constellation
    MAX_FEATURES = 200
    FEATURE_QUALITY = 0.01
    MIN_FEATURE_DISTANCE = 15

    # Color extraction
    NUM_COLORS = 6

    # HUD
    GRID_DIVISIONS = 8


# =============================================================================
# LAYOUT CALCULATION
# =============================================================================

def calculate_layout(img_width, img_height, config=Config):
    """
    Calculate canvas size and component positions based on image aspect ratio.
    Returns: (canvas_w, canvas_h, img_x, img_y, img_w, img_h, panel_info, layout_mode)
    """
    aspect = img_width / img_height

    # Determine layout mode
    if config.AUTO_LAYOUT:
        if aspect > config.PANORAMA_THRESHOLD:
            layout_mode = LayoutMode.PANORAMA
        elif aspect > config.LANDSCAPE_THRESHOLD:
            layout_mode = LayoutMode.LANDSCAPE
        elif aspect < config.PORTRAIT_THRESHOLD:
            layout_mode = LayoutMode.PORTRAIT
        else:
            layout_mode = LayoutMode.LANDSCAPE  # Default for square-ish
    else:
        layout_mode = LayoutMode.LANDSCAPE

    # Apply scale factor
    target_width = int(img_width * config.SCALE_FACTOR)
    target_height = int(img_height * config.SCALE_FACTOR)

    # Clamp to min/max
    if target_width > config.MAX_OUTPUT_WIDTH:
        scale = config.MAX_OUTPUT_WIDTH / target_width
        target_width = config.MAX_OUTPUT_WIDTH
        target_height = int(target_height * scale)
    elif target_width < config.MIN_OUTPUT_WIDTH:
        scale = config.MIN_OUTPUT_WIDTH / target_width
        target_width = config.MIN_OUTPUT_WIDTH
        target_height = int(target_height * scale)

    # Calculate layout based on mode
    if layout_mode == LayoutMode.LANDSCAPE:
        # Side panel layout (original)
        panel_width = int(target_width * config.PANEL_SIZE_LANDSCAPE)
        canvas_width = target_width + panel_width + config.PADDING * 3
        canvas_height = target_height + config.PADDING * 2

        img_x = config.PADDING
        img_y = config.PADDING

        panel_info = {
            'x': img_x + target_width + config.PADDING,
            'y': img_y,
            'width': panel_width,
            'height': target_height,
            'orientation': 'vertical'
        }

    elif layout_mode == LayoutMode.PORTRAIT:
        # Bottom panel layout
        panel_height = int(target_height * config.PANEL_SIZE_PORTRAIT)
        canvas_width = target_width + config.PADDING * 2
        canvas_height = target_height + panel_height + config.PADDING * 3

        img_x = config.PADDING
        img_y = config.PADDING

        panel_info = {
            'x': img_x,
            'y': img_y + target_height + config.PADDING,
            'width': target_width,
            'height': panel_height,
            'orientation': 'horizontal'
        }

    else:  # PANORAMA
        # Top panel layout
        panel_height = int(target_height * config.PANEL_SIZE_PANORAMA)
        canvas_width = target_width + config.PADDING * 2
        canvas_height = target_height + panel_height + config.PADDING * 3

        img_x = config.PADDING
        img_y = config.PADDING + panel_height + config.PADDING

        panel_info = {
            'x': img_x,
            'y': config.PADDING,
            'width': target_width,
            'height': panel_height,
            'orientation': 'horizontal'
        }

    return (canvas_width, canvas_height, img_x, img_y,
            target_width, target_height, panel_info, layout_mode)


# =============================================================================
# LAYER FUNCTIONS (Same as original, with minor adjustments)
# =============================================================================

def generate_contours(gray, config=Config):
    """Extract topographic-style contour lines from luminance."""
    h, w = gray.shape
    smoothed = gaussian_filter(gray.astype(float), sigma=config.CONTOUR_SMOOTHING)

    contours_all = []
    levels = np.linspace(15, 240, config.CONTOUR_LEVELS)

    for level in levels:
        binary = (smoothed > level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.arcLength(c, True) > 30]
        contours_all.append((level, contours))

    return contours_all


def draw_contours(canvas, contours_all, offset, scale, config=Config):
    """Draw contour lines with varying opacity."""
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for i, (level, contours) in enumerate(contours_all):
        alpha = int((80 + (level / 255) * 100) * config.OVERLAY_ALPHA_BOOST)
        alpha = min(alpha, 255)
        color = (config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha)

        for contour in contours:
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue

            scaled = [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1]))
                      for p in points]

            if len(scaled) > 2:
                draw.line(scaled + [scaled[0]], fill=color, width=config.CONTOUR_LINE_WIDTH)

    return canvas


def compute_gradient_field(gray, config=Config):
    """Compute gradient vectors using Sobel operators."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    h, w = gray.shape
    spacing = config.VECTOR_GRID_SPACING

    vectors = []
    for y in range(spacing, h - spacing, spacing):
        for x in range(spacing, w - spacing, spacing):
            mag = magnitude[y, x]
            ang = angle[y, x]
            vectors.append((x, y, mag, ang))

    return vectors, magnitude.max()


def draw_vector_field(canvas, vectors, max_mag, offset, scale, config=Config):
    """Draw gradient arrows."""
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for x, y, mag, ang in vectors:
        if mag < max_mag * config.VECTOR_MIN_THRESHOLD:
            continue

        norm_mag = (mag / max_mag) * config.VECTOR_SCALE
        norm_mag = max(norm_mag, 4)

        sx = int(x * scale + offset[0])
        sy = int(y * scale + offset[1])
        ex = int(sx + np.cos(ang) * norm_mag)
        ey = int(sy + np.sin(ang) * norm_mag)

        alpha = int((50 + (mag / max_mag) * 150) * config.OVERLAY_ALPHA_BOOST)
        alpha = min(alpha, 255)
        color = (config.SECONDARY[0], config.SECONDARY[1], config.SECONDARY[2], alpha)

        draw.line([(sx, sy), (ex, ey)], fill=color, width=1)
        draw.ellipse([sx-1, sy-1, sx+1, sy+1], fill=color)

    return canvas


def compute_fft_spectrum(gray):
    """Compute 2D FFT magnitude spectrum."""
    h, w = gray.shape
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)

    windowed = gray.astype(float) * window
    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    log_magnitude = np.log1p(magnitude)
    log_magnitude = log_magnitude / log_magnitude.max()

    gamma = 0.3
    log_magnitude = np.power(log_magnitude, gamma)

    p_low, p_high = np.percentile(log_magnitude, [1, 99])
    log_magnitude = np.clip((log_magnitude - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return (log_magnitude * 255).astype(np.uint8)


def extract_dominant_colors(image, config=Config):
    """Extract dominant colors using K-means."""
    pixels = np.array(image).reshape(-1, 3)

    sample_size = min(10000, len(pixels))
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    sample = pixels[indices]

    kmeans = KMeans(n_clusters=config.NUM_COLORS, random_state=42, n_init=10)
    kmeans.fit(sample)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    percentages = counts / counts.sum()

    results = []
    for color, pct in zip(colors, percentages):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        results.append((tuple(color), pct, lum))

    results.sort(key=lambda x: x[2])
    return [(r[0], r[1]) for r in results]


def detect_features(gray, config=Config):
    """Detect corner features using Harris detector."""
    h, w = gray.shape

    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    threshold = config.FEATURE_QUALITY * harris.max()

    grid_size = int(np.sqrt(config.MAX_FEATURES))
    cell_h = h // grid_size
    cell_w = w // grid_size

    points = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y_start = gy * cell_h
            y_end = min((gy + 1) * cell_h, h)
            x_start = gx * cell_w
            x_end = min((gx + 1) * cell_w, w)

            cell = harris[y_start:y_end, x_start:x_end]

            if cell.max() > threshold * 0.5:
                local_coords = np.unravel_index(cell.argmax(), cell.shape)
                points.append([x_start + local_coords[1], y_start + local_coords[0]])
            else:
                cx = x_start + cell_w // 2 + np.random.randint(-cell_w//4, cell_w//4)
                cy = y_start + cell_h // 2 + np.random.randint(-cell_h//4, cell_h//4)
                cx = np.clip(cx, x_start, x_end - 1)
                cy = np.clip(cy, y_start, y_end - 1)
                points.append([cx, cy])

    return np.array(points) if points else np.array([]).reshape(0, 2)


def compute_delaunay(points):
    """Compute Delaunay triangulation."""
    if len(points) < 4:
        return None
    try:
        return Delaunay(points)
    except:
        return None


def draw_constellation(canvas, points, triangulation, offset, scale, config=Config):
    """Draw feature points and Delaunay mesh."""
    draw = ImageDraw.Draw(canvas, 'RGBA')

    if len(points) == 0:
        return canvas

    scaled_points = [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1]))
                     for p in points]

    if triangulation is not None:
        edges = set()
        for simplex in triangulation.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges.add(edge)

        for i, j in edges:
            if i >= len(scaled_points) or j >= len(scaled_points):
                continue
            p1, p2 = scaled_points[i], scaled_points[j]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            alpha = int(max(20, min(90, int(200 - dist * 0.4))) * config.OVERLAY_ALPHA_BOOST)
            draw.line([p1, p2], fill=(config.TERTIARY[0], config.TERTIARY[1],
                                       config.TERTIARY[2], alpha), width=1)

    for px, py in scaled_points:
        draw.ellipse([px-5, py-5, px+5, py+5],
                     fill=(config.TERTIARY[0], config.TERTIARY[1], config.TERTIARY[2], 30))
        draw.ellipse([px-3, py-3, px+3, py+3],
                     fill=(config.TERTIARY[0], config.TERTIARY[1], config.TERTIARY[2], 60))
        draw.ellipse([px-2, py-2, px+2, py+2], fill=config.TERTIARY)

    return canvas


# =============================================================================
# ADAPTIVE PANEL RENDERING
# =============================================================================

def draw_analysis_panels(canvas, spectrum, colors, panel_info, stats, config=Config):
    """Draw FFT, color spectra, and stats in adaptive layout."""
    draw = ImageDraw.Draw(canvas, 'RGBA')

    px, py = panel_info['x'], panel_info['y']
    pw, ph = panel_info['width'], panel_info['height']

    if panel_info['orientation'] == 'vertical':
        # Vertical layout (side panel)
        # FFT spectrum (top)
        fft_h = int(pw * 0.75)
        draw_fft_inset(canvas, spectrum, (px, py), (pw, fft_h), config)

        # Color spectra (middle)
        color_y = py + fft_h + 40
        color_h = 180
        draw_color_spectra(canvas, colors, (px, color_y), (pw, color_h), config)

        # Stats (bottom)
        stats_y = color_y + color_h + 40
        draw.text((px, stats_y), "─── ANALYSIS METRICS ───", fill=config.DIM)
        for i, (label, value) in enumerate(stats.items()):
            draw.text((px, stats_y + 20 + i * 18), f"{label}: {value}", fill=config.DIM)

    else:
        # Horizontal layout (top/bottom panel)
        section_w = pw // 3

        # FFT (left third)
        fft_w = section_w - 20
        fft_h = min(ph - 60, int(fft_w * 0.75))
        draw_fft_inset(canvas, spectrum, (px, py), (fft_w, fft_h), config)

        # Colors (middle third)
        color_x = px + section_w
        color_w = section_w - 20
        color_h = ph - 60
        draw_color_spectra(canvas, colors, (color_x, py), (color_w, color_h), config)

        # Stats (right third)
        stats_x = px + section_w * 2
        draw.text((stats_x, py), "─── ANALYSIS METRICS ───", fill=config.DIM)
        for i, (label, value) in enumerate(stats.items()):
            draw.text((stats_x, py + 20 + i * 18), f"{label}: {value}", fill=config.DIM)


def draw_fft_inset(canvas, spectrum, position, size, config=Config):
    """Draw FFT spectrum inset."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    spec_img = Image.fromarray(spectrum)
    spec_img = spec_img.resize((w - 4, h - 4), Image.Resampling.LANCZOS)
    spec_array = np.array(spec_img)

    colored = np.zeros((spec_array.shape[0], spec_array.shape[1], 4), dtype=np.uint8)
    intensity = spec_array.astype(float) / 255.0
    colored[:, :, 0] = (intensity * config.PRIMARY[0]).astype(np.uint8)
    colored[:, :, 1] = (intensity * config.PRIMARY[1]).astype(np.uint8)
    colored[:, :, 2] = (intensity * config.PRIMARY[2]).astype(np.uint8)
    colored[:, :, 3] = (intensity * 255).astype(np.uint8)

    base = np.zeros((spec_array.shape[0], spec_array.shape[1], 4), dtype=np.uint8)
    base[:, :, 0:3] = 245
    base[:, :, 3] = 255

    base_img = Image.fromarray(base, 'RGBA')
    spec_colored = Image.fromarray(colored, 'RGBA')

    draw.rectangle([x, y, x + w, y + h], fill=(245, 248, 250, 255), outline=config.PRIMARY, width=1)

    combined = Image.alpha_composite(base_img, spec_colored)
    canvas.paste(combined, (x + 2, y + 2))

    draw.text((x + 4, y + h + 4), "FREQ SPECTRUM [FFT²]", fill=config.DIM)

    return canvas


def draw_color_spectra(canvas, colors, position, size, config=Config):
    """Draw color spectrum bars."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    draw.rectangle([x, y, x + w, y + h], fill=(250, 250, 250, 255), outline=config.PRIMARY, width=1)

    bar_height = (h - 20) // len(colors)

    for i, (color, pct) in enumerate(colors):
        by = y + 10 + i * bar_height
        bar_width = int((w - 20) * pct * 2)
        bar_width = min(bar_width, w - 20)

        draw.rectangle([x + 12, by + 2, x + 12 + bar_width, by + bar_height - 2],
                      fill=(200, 200, 200, 100))
        draw.rectangle([x + 10, by, x + 10 + bar_width, by + bar_height - 4],
                      fill=(color[0], color[1], color[2], 255),
                      outline=(color[0]//2, color[1]//2, color[2]//2, 100), width=1)

        hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        draw.text((x + w - 70, by), hex_code, fill=config.DIM)

    draw.text((x + 4, y + h + 4), "CHROMATIC EMISSION", fill=config.DIM)

    return canvas


def draw_hud_frame(canvas, image_bounds, original_size, layout_mode, config=Config):
    """Draw HUD overlays adapted to layout."""
    draw = ImageDraw.Draw(canvas, 'RGBA')

    x, y, w, h = image_bounds
    orig_w, orig_h = original_size

    # Scan grid
    grid_lines = 16
    for i in range(1, grid_lines):
        lx = x + (w * i // grid_lines)
        alpha = 40 if i % 4 != 0 else 70
        draw.line([(lx, y), (lx, y + h)],
                 fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)

        ly = y + (h * i // grid_lines)
        draw.line([(x, ly), (x + w, ly)],
                 fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)

    # Corner brackets
    bracket_len = 30
    bracket_color = config.PRIMARY

    draw.line([(x, y), (x + bracket_len, y)], fill=bracket_color, width=2)
    draw.line([(x, y), (x, y + bracket_len)], fill=bracket_color, width=2)
    draw.line([(x + w, y), (x + w - bracket_len, y)], fill=bracket_color, width=2)
    draw.line([(x + w, y), (x + w, y + bracket_len)], fill=bracket_color, width=2)
    draw.line([(x, y + h), (x + bracket_len, y + h)], fill=bracket_color, width=2)
    draw.line([(x, y + h), (x, y + h - bracket_len)], fill=bracket_color, width=2)
    draw.line([(x + w, y + h), (x + w - bracket_len, y + h)], fill=bracket_color, width=2)
    draw.line([(x + w, y + h), (x + w, y + h - bracket_len)], fill=bracket_color, width=2)

    # Coordinate ticks
    divisions = config.GRID_DIVISIONS
    for i in range(divisions + 1):
        tx = x + (w * i // divisions)
        coord_x = orig_w * i // divisions

        draw.line([(tx, y - 8), (tx, y - 2)], fill=config.DIM, width=1)
        draw.line([(tx, y + h + 2), (tx, y + h + 8)], fill=config.DIM, width=1)

        if i % 2 == 0:
            draw.text((tx - 15, y - 22), f"{coord_x:04d}", fill=config.DIM)

        ty = y + (h * i // divisions)
        coord_y = orig_h * i // divisions

        draw.line([(x - 8, ty), (x - 2, ty)], fill=config.DIM, width=1)
        draw.line([(x + w + 2, ty), (x + w + 8, ty)], fill=config.DIM, width=1)

        if i % 2 == 0:
            draw.text((x - 45, ty - 6), f"{coord_y:04d}", fill=config.DIM)

    # Technical readouts
    canvas_w, canvas_h = canvas.size

    draw.text((15, 15), f"SOURCE: {orig_w}×{orig_h}px", fill=config.PRIMARY)
    draw.text((15, 32), f"MODE: {layout_mode.value.upper()}", fill=config.DIM)

    draw.text((canvas_w - 200, 15), "SCAN COMPLETE", fill=config.SECONDARY)
    draw.text((canvas_w - 200, 32), f"GRID: {divisions}×{divisions}", fill=config.DIM)

    # Crosshair
    cx, cy = x + w // 2, y + h // 2
    cross_size = 20
    cross_color = (config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], 120)
    draw.line([(cx - cross_size, cy), (cx - 5, cy)], fill=cross_color, width=1)
    draw.line([(cx + 5, cy), (cx + cross_size, cy)], fill=cross_color, width=1)
    draw.line([(cx, cy - cross_size), (cx, cy - 5)], fill=cross_color, width=1)
    draw.line([(cx, cy + 5), (cx, cy + cross_size)], fill=cross_color, width=1)

    return canvas


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_image(input_path, output_path=None, config=Config):
    """
    Enhanced processing pipeline with resolution preservation and adaptive layout.
    """
    # Validate input
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    print(f"Loading: {input_path}")

    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    orig_w, orig_h = img.size
    print(f"  Original size: {orig_w}×{orig_h} (aspect: {orig_w/orig_h:.2f})")

    # Calculate layout
    canvas_w, canvas_h, img_x, img_y, img_w, img_h, panel_info, layout_mode = \
        calculate_layout(orig_w, orig_h, config)

    print(f"  Layout mode: {layout_mode.value}")
    print(f"  Output size: {canvas_w}×{canvas_h}")
    print(f"  Image area: {img_w}×{img_h}")

    # Create canvas
    canvas = Image.new('RGBA', (canvas_w, canvas_h), config.BACKGROUND)

    # Resize image
    img_resized = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply base image with optional fade
    base_img = Image.fromarray(img_array)
    if config.BASE_IMAGE_FADE > 0:
        base_img = Image.blend(base_img,
                               Image.new('RGB', base_img.size, (255, 255, 255)),
                               alpha=config.BASE_IMAGE_FADE)
    canvas.paste(base_img, (img_x, img_y))

    scale = 1.0
    offset = (img_x, img_y)

    # Generate layers
    print("  [1/6] Computing luminance contours...")
    contours = generate_contours(gray, config)
    canvas = draw_contours(canvas, contours, offset, scale, config)

    print("  [2/6] Computing gradient vector field...")
    vectors, max_mag = compute_gradient_field(gray, config)
    canvas = draw_vector_field(canvas, vectors, max_mag, offset, scale, config)

    print("  [3/6] Computing frequency spectrum...")
    spectrum = compute_fft_spectrum(gray)

    print("  [4/6] Extracting dominant colors...")
    colors = extract_dominant_colors(img_resized, config)

    print("  [5/6] Detecting features and triangulation...")
    features = detect_features(gray, config)
    triangulation = compute_delaunay(features)
    canvas = draw_constellation(canvas, features, triangulation, offset, scale, config)

    print("  [6/6] Drawing HUD and panels...")
    image_bounds = (img_x, img_y, img_w, img_h)
    canvas = draw_hud_frame(canvas, image_bounds, (orig_w, orig_h), layout_mode, config)

    # Draw analysis panels
    stats = {
        "Contour levels": config.CONTOUR_LEVELS,
        "Feature points": len(features),
        "Gradient samples": len(vectors),
        "Color clusters": config.NUM_COLORS,
    }
    draw_analysis_panels(canvas, spectrum, colors, panel_info, stats, config)

    # Add legend
    draw = ImageDraw.Draw(canvas, 'RGBA')
    legend_y = canvas_h - 30
    legend_items = [
        (config.PRIMARY, "CONTOURS"),
        (config.SECONDARY, "GRADIENT"),
        (config.TERTIARY, "FEATURES"),
    ]
    lx = config.PADDING
    for color, label in legend_items:
        draw.rectangle([lx, legend_y, lx + 12, legend_y + 12], fill=color)
        draw.text((lx + 18, legend_y - 2), label, fill=config.DIM)
        lx += 120

    # Save
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_survey_enhanced{input_file.suffix}"

    final = Image.new('RGB', canvas.size, (255, 255, 255))
    final.paste(canvas, mask=canvas.split()[3])

    final.save(output_path, quality=95)
    print(f"\nSaved: {output_path}")

    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    def select_file():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select Image for Orbital Survey",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path

    def select_output():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.asksaveasfilename(
            title="Save Survey Result As",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        root.destroy()
        return file_path

    # Parse command line
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("Select an image file...")
        input_path = select_file()

        if not input_path:
            print("No file selected. Exiting.")
            sys.exit(0)

        print(f"Selected: {input_path}")
        print("Select output location (or cancel for auto-naming)...")
        output_path = select_output()

        if not output_path:
            output_path = None
            print("Using auto-generated output filename.")

    try:
        result = process_image(input_path, output_path)
        print(f"\n✓ Done! Output: {result}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
