"""
ORBITAL SURVEY PROTOCOL - V2
=============================
Enhanced visuals with cleaner design, better readability, and improved panels.

V2 IMPROVEMENTS:
- Larger side panel (40% for better readability)
- Percentage labels on color bars
- Radial FFT spectrum visualization
- Cleaner typography and spacing
- Improved visual hierarchy
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from pathlib import Path
import sys
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class LayoutMode(Enum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    PANORAMA = "panorama"


class Config:
    # Resolution Control
    SCALE_FACTOR = 1.0
    MAX_OUTPUT_WIDTH = 3840
    MIN_OUTPUT_WIDTH = 1280

    # Layout
    AUTO_LAYOUT = True
    LANDSCAPE_THRESHOLD = 1.3
    PORTRAIT_THRESHOLD = 0.75
    PANORAMA_THRESHOLD = 2.5

    # Panel sizing - INCREASED for better readability
    PANEL_SIZE_LANDSCAPE = 0.40  # 40% instead of 30%
    PANEL_SIZE_PORTRAIT = 0.30   # 30% instead of 25%
    PANEL_SIZE_PANORAMA = 0.25   # 25% instead of 20%

    # Padding
    PADDING = 80
    PANEL_PADDING = 30
    SECTION_SPACING = 40

    # Colors - refined palette
    PRIMARY = (0, 180, 160, 255)       # Teal
    SECONDARY = (220, 80, 40, 255)     # Red-orange
    TERTIARY = (120, 60, 200, 255)     # Purple
    ACCENT = (255, 180, 0, 255)        # Gold
    DIM = (80, 80, 80, 255)            # Darker gray for better contrast
    LIGHT_GRAY = (200, 200, 200, 255)  # For dividers
    BACKGROUND = (255, 255, 255, 255)

    # Typography
    FONT_SIZE_TITLE = 16
    FONT_SIZE_LABEL = 12
    FONT_SIZE_VALUE = 14

    # Overlay intensity
    BASE_IMAGE_FADE = 0.12
    OVERLAY_ALPHA_BOOST = 1.0

    # Analysis parameters
    CONTOUR_LEVELS = 20
    CONTOUR_SMOOTHING = 3.0
    CONTOUR_LINE_WIDTH = 2
    VECTOR_GRID_SPACING = 30
    VECTOR_SCALE = 20
    VECTOR_MIN_THRESHOLD = 0.03
    MAX_FEATURES = 200
    FEATURE_QUALITY = 0.01
    NUM_COLORS = 6
    GRID_DIVISIONS = 8


# =============================================================================
# LAYOUT CALCULATION
# =============================================================================

def calculate_layout(img_width, img_height, config=Config):
    """Calculate adaptive layout based on aspect ratio."""
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
            layout_mode = LayoutMode.LANDSCAPE
    else:
        layout_mode = LayoutMode.LANDSCAPE

    # Apply scale factor with constraints
    target_width = int(img_width * config.SCALE_FACTOR)
    target_height = int(img_height * config.SCALE_FACTOR)

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
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def generate_contours(gray, config=Config):
    """Extract topographic contour lines."""
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
    """Draw contour lines."""
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
    """Compute gradient vectors."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    h, w = gray.shape
    spacing = config.VECTOR_GRID_SPACING

    vectors = []
    for y in range(spacing, h - spacing, spacing):
        for x in range(spacing, w - spacing, spacing):
            vectors.append((x, y, magnitude[y, x], angle[y, x]))

    return vectors, magnitude.max()


def draw_vector_field(canvas, vectors, max_mag, offset, scale, config=Config):
    """Draw gradient arrows."""
    draw = ImageDraw.Draw(canvas, 'RGBA')

    for x, y, mag, ang in vectors:
        if mag < max_mag * config.VECTOR_MIN_THRESHOLD:
            continue

        norm_mag = max((mag / max_mag) * config.VECTOR_SCALE, 4)

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
    """Compute 2D FFT spectrum."""
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray.astype(float) * window

    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    log_magnitude = np.log1p(magnitude)
    log_magnitude = log_magnitude / log_magnitude.max()
    log_magnitude = np.power(log_magnitude, 0.3)

    p_low, p_high = np.percentile(log_magnitude, [1, 99])
    log_magnitude = np.clip((log_magnitude - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return (log_magnitude * 255).astype(np.uint8)


def extract_dominant_colors(image, config=Config):
    """Extract dominant colors with K-means."""
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
    """Detect corner features."""
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
    """Draw feature constellation."""
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
# ENHANCED VISUALIZATION PANELS
# =============================================================================

def draw_radial_fft(canvas, spectrum, position, size, config=Config):
    """Draw FFT as radial/circular visualization - more visually interesting."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Create background
    draw.rectangle([x, y, x + w, y + h], fill=(248, 250, 252, 255), outline=config.PRIMARY, width=2)

    # Title
    title_y = y + 10
    draw.text((x + 10, title_y), "FREQUENCY SPECTRUM [2D-FFT]", fill=config.PRIMARY,
              font=None)

    # Convert spectrum to circular coordinates
    spec_size = min(w - 40, h - 80)
    center_x = x + w // 2
    center_y = y + 60 + spec_size // 2

    # Resize spectrum
    spec_img = Image.fromarray(spectrum)
    spec_img = spec_img.resize((spec_size, spec_size), Image.Resampling.LANCZOS)
    spec_array = np.array(spec_img)

    # Create radial visualization
    radius = spec_size // 2

    # Draw concentric circles with FFT data
    num_rings = 40
    for ring in range(num_rings):
        r = int((ring / num_rings) * radius)
        if r == 0:
            continue

        # Sample the spectrum at this radius
        num_samples = max(8, int(2 * np.pi * r))
        for i in range(num_samples):
            angle = (i / num_samples) * 2 * np.pi

            # Sample position in spectrum
            sx = int(spec_size // 2 + r * np.cos(angle))
            sy = int(spec_size // 2 + r * np.sin(angle))

            if 0 <= sx < spec_size and 0 <= sy < spec_size:
                intensity = spec_array[sy, sx] / 255.0

                # Calculate display position
                px = int(center_x + r * np.cos(angle))
                py = int(center_y + r * np.sin(angle))

                # Color based on intensity - gradient from teal to gold
                if intensity > 0.1:
                    alpha = int(intensity * 200)
                    # Blend primary and accent based on radius
                    t = ring / num_rings
                    color = (
                        int(config.PRIMARY[0] * (1-t) + config.ACCENT[0] * t),
                        int(config.PRIMARY[1] * (1-t) + config.ACCENT[1] * t),
                        int(config.PRIMARY[2] * (1-t) + config.ACCENT[2] * t),
                        alpha
                    )
                    draw.ellipse([px-1, py-1, px+1, py+1], fill=color)

    # Draw center marker
    draw.ellipse([center_x-3, center_y-3, center_x+3, center_y+3], fill=config.SECONDARY)

    # Concentric reference circles
    for r in [radius//4, radius//2, 3*radius//4]:
        draw.ellipse([center_x-r, center_y-r, center_x+r, center_y+r],
                    outline=(config.LIGHT_GRAY[0], config.LIGHT_GRAY[1], config.LIGHT_GRAY[2], 100),
                    width=1)

    # Labels
    draw.text((x + 10, y + h - 20), "Center: DC Component", fill=config.DIM, font=None)
    draw.text((x + w - 130, y + h - 20), "Edge: High Freq", fill=config.DIM, font=None)

    return canvas


def draw_color_analysis(canvas, colors, position, size, config=Config):
    """Draw color analysis with clear percentages."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Background
    draw.rectangle([x, y, x + w, y + h], fill=(248, 250, 252, 255), outline=config.PRIMARY, width=2)

    # Title
    title_y = y + 10
    draw.text((x + 10, title_y), "CHROMATIC ANALYSIS", fill=config.PRIMARY, font=None)

    # Color bars with percentages
    bar_start_y = y + 40
    bar_height = (h - 60) // len(colors)
    max_bar_width = w - 140

    for i, (color, pct) in enumerate(colors):
        by = bar_start_y + i * bar_height
        bar_width = int(max_bar_width * pct)

        # Background bar (full width, light)
        draw.rectangle([x + 10, by, x + 10 + max_bar_width, by + bar_height - 8],
                      fill=(235, 235, 235, 255), outline=None)

        # Actual color bar
        draw.rectangle([x + 10, by, x + 10 + bar_width, by + bar_height - 8],
                      fill=(color[0], color[1], color[2], 255),
                      outline=(color[0]//2, color[1]//2, color[2]//2, 255), width=1)

        # Percentage text - CLEAR AND VISIBLE
        pct_text = f"{pct*100:.1f}%"
        # Position text at the end of the bar
        text_x = x + 10 + max_bar_width + 10
        draw.text((text_x, by + 2), pct_text, fill=config.DIM, font=None)

        # Hex code
        hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        draw.text((text_x, by + 18), hex_code, fill=(180, 180, 180, 255), font=None)

    return canvas


def draw_stats_panel(canvas, stats, position, size, config=Config):
    """Draw statistics panel with clean layout."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Background
    draw.rectangle([x, y, x + w, y + h], fill=(248, 250, 252, 255), outline=config.PRIMARY, width=2)

    # Title
    draw.text((x + 10, y + 10), "ANALYSIS METRICS", fill=config.PRIMARY, font=None)

    # Stats
    stat_y = y + 40
    for label, value in stats.items():
        # Label
        draw.text((x + 15, stat_y), label, fill=config.DIM, font=None)

        # Value (right-aligned)
        value_text = str(value)
        draw.text((x + w - 50, stat_y), value_text, fill=config.SECONDARY, font=None)

        stat_y += 25

    return canvas


def draw_analysis_panels(canvas, spectrum, colors, panel_info, stats, config=Config):
    """Draw all analysis panels with improved layout."""
    px, py = panel_info['x'], panel_info['y']
    pw, ph = panel_info['width'], panel_info['height']

    if panel_info['orientation'] == 'vertical':
        # Vertical layout (side panel) - REDESIGNED
        spacing = config.SECTION_SPACING

        # FFT - larger and more interesting
        fft_h = int(pw * 1.0)  # Square-ish
        draw_radial_fft(canvas, spectrum, (px, py), (pw, fft_h), config)

        # Color analysis
        color_y = py + fft_h + spacing
        color_h = 220
        draw_color_analysis(canvas, colors, (px, color_y), (pw, color_h), config)

        # Stats
        stats_y = color_y + color_h + spacing
        stats_h = ph - (stats_y - py) - 10
        if stats_h > 80:
            draw_stats_panel(canvas, stats, (px, stats_y), (pw, stats_h), config)

    else:
        # Horizontal layout
        section_w = pw // 3
        spacing = 20

        # FFT (left)
        fft_w = section_w - spacing
        fft_h = ph - 20
        draw_radial_fft(canvas, spectrum, (px, py), (fft_w, fft_h), config)

        # Colors (middle)
        color_x = px + section_w
        color_w = section_w - spacing
        color_h = ph - 20
        draw_color_analysis(canvas, colors, (color_x, py), (color_w, color_h), config)

        # Stats (right)
        stats_x = px + section_w * 2
        stats_w = section_w - spacing
        stats_h = ph - 20
        draw_stats_panel(canvas, stats, (stats_x, py), (stats_w, stats_h), config)


def draw_hud_frame(canvas, image_bounds, original_size, layout_mode, config=Config):
    """Draw HUD frame with cleaner design."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y, w, h = image_bounds
    orig_w, orig_h = original_size

    # Scan grid - cleaner
    grid_lines = 12  # Reduced for cleaner look
    for i in range(1, grid_lines):
        lx = x + (w * i // grid_lines)
        alpha = 25 if i % 3 != 0 else 50
        draw.line([(lx, y), (lx, y + h)],
                 fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)

        ly = y + (h * i // grid_lines)
        draw.line([(x, ly), (x + w, ly)],
                 fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)

    # Corner brackets - thicker and cleaner
    bracket_len = 40
    draw.line([(x, y), (x + bracket_len, y)], fill=config.PRIMARY, width=3)
    draw.line([(x, y), (x, y + bracket_len)], fill=config.PRIMARY, width=3)
    draw.line([(x + w, y), (x + w - bracket_len, y)], fill=config.PRIMARY, width=3)
    draw.line([(x + w, y), (x + w, y + bracket_len)], fill=config.PRIMARY, width=3)
    draw.line([(x, y + h), (x + bracket_len, y + h)], fill=config.PRIMARY, width=3)
    draw.line([(x, y + h), (x, y + h - bracket_len)], fill=config.PRIMARY, width=3)
    draw.line([(x + w, y + h), (x + w - bracket_len, y + h)], fill=config.PRIMARY, width=3)
    draw.line([(x + w, y + h), (x + w, y + h - bracket_len)], fill=config.PRIMARY, width=3)

    # Simplified coordinate ticks
    divisions = config.GRID_DIVISIONS
    for i in range(0, divisions + 1, 2):  # Every other tick
        tx = x + (w * i // divisions)
        coord_x = orig_w * i // divisions

        draw.line([(tx, y - 8), (tx, y - 2)], fill=config.DIM, width=2)
        draw.text((tx - 15, y - 24), f"{coord_x:04d}", fill=config.DIM)

        ty = y + (h * i // divisions)
        coord_y = orig_h * i // divisions

        draw.line([(x - 8, ty), (x - 2, ty)], fill=config.DIM, width=2)
        draw.text((x - 50, ty - 6), f"{coord_y:04d}", fill=config.DIM)

    # Clean header
    canvas_w, canvas_h = canvas.size
    draw.text((15, 15), f"ORBITAL SURVEY v2.0", fill=config.PRIMARY, font=None)
    draw.text((15, 35), f"Source: {orig_w}×{orig_h}px", fill=config.DIM, font=None)

    draw.text((canvas_w - 180, 15), "SCAN COMPLETE", fill=config.SECONDARY, font=None)
    draw.text((canvas_w - 180, 35), f"Mode: {layout_mode.value.upper()}", fill=config.DIM, font=None)

    # Crosshair
    cx, cy = x + w // 2, y + h // 2
    cross_size = 25
    draw.line([(cx - cross_size, cy), (cx - 8, cy)], fill=config.PRIMARY, width=2)
    draw.line([(cx + 8, cy), (cx + cross_size, cy)], fill=config.PRIMARY, width=2)
    draw.line([(cx, cy - cross_size), (cx, cy - 8)], fill=config.PRIMARY, width=2)
    draw.line([(cx, cy + 8), (cx, cy + cross_size)], fill=config.PRIMARY, width=2)
    draw.ellipse([cx-4, cy-4, cx+4, cy+4], outline=config.PRIMARY, width=2)

    return canvas


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_image(input_path, output_path=None, config=Config):
    """Process image with enhanced visuals."""

    # Validate input
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading: {input_path}")

    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    orig_w, orig_h = img.size
    print(f"  Original: {orig_w}×{orig_h} (aspect: {orig_w/orig_h:.2f})")

    # Calculate layout
    canvas_w, canvas_h, img_x, img_y, img_w, img_h, panel_info, layout_mode = \
        calculate_layout(orig_w, orig_h, config)

    print(f"  Layout: {layout_mode.value}")
    print(f"  Canvas: {canvas_w}×{canvas_h}")
    print(f"  Image area: {img_w}×{img_h}")
    print(f"  Panel: {panel_info['width']}×{panel_info['height']}")

    # Create canvas
    canvas = Image.new('RGBA', (canvas_w, canvas_h), config.BACKGROUND)

    # Resize image
    img_resized = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply base image
    base_img = Image.fromarray(img_array)
    if config.BASE_IMAGE_FADE > 0:
        base_img = Image.blend(base_img,
                               Image.new('RGB', base_img.size, (255, 255, 255)),
                               alpha=config.BASE_IMAGE_FADE)
    canvas.paste(base_img, (img_x, img_y))

    scale = 1.0
    offset = (img_x, img_y)

    # Generate layers
    print("  [1/6] Computing contours...")
    contours = generate_contours(gray, config)
    canvas = draw_contours(canvas, contours, offset, scale, config)

    print("  [2/6] Computing gradient field...")
    vectors, max_mag = compute_gradient_field(gray, config)
    canvas = draw_vector_field(canvas, vectors, max_mag, offset, scale, config)

    print("  [3/6] Computing FFT spectrum...")
    spectrum = compute_fft_spectrum(gray)

    print("  [4/6] Extracting colors...")
    colors = extract_dominant_colors(img_resized, config)

    print("  [5/6] Detecting features...")
    features = detect_features(gray, config)
    triangulation = compute_delaunay(features)
    canvas = draw_constellation(canvas, features, triangulation, offset, scale, config)

    print("  [6/6] Drawing HUD and panels...")
    image_bounds = (img_x, img_y, img_w, img_h)
    canvas = draw_hud_frame(canvas, image_bounds, (orig_w, orig_h), layout_mode, config)

    # Draw enhanced panels
    stats = {
        "Contours": config.CONTOUR_LEVELS,
        "Features": len(features),
        "Vectors": len(vectors),
        "Colors": config.NUM_COLORS,
    }
    draw_analysis_panels(canvas, spectrum, colors, panel_info, stats, config)

    # Legend
    draw = ImageDraw.Draw(canvas, 'RGBA')
    legend_y = canvas_h - 35
    legend_items = [
        (config.PRIMARY, "CONTOURS"),
        (config.SECONDARY, "GRADIENT"),
        (config.TERTIARY, "FEATURES"),
    ]
    lx = config.PADDING
    for color, label in legend_items:
        draw.rectangle([lx, legend_y, lx + 16, legend_y + 16], fill=color, outline=config.DIM, width=1)
        draw.text((lx + 22, legend_y + 2), label, fill=config.DIM)
        lx += 140

    # Save
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_survey_v2{input_file.suffix}"

    final = Image.new('RGB', canvas.size, (255, 255, 255))
    final.paste(canvas, mask=canvas.split()[3])
    final.save(output_path, quality=95)

    print(f"\n✓ Saved: {output_path}")
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

    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("Select an image file...")
        input_path = select_file()
        if not input_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        output_path = None

    try:
        result = process_image(input_path, output_path)
        print(f"\n✓ Done!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
