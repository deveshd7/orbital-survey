"""
ORBITAL SURVEY PROTOCOL - V3
=============================
Analysis-focused design: Panels are the star, image is reference.

V3 IMPROVEMENTS:
- Smaller image (50% of canvas) - viewer sees analysis, not every pixel
- Color analysis is now the primary focus with larger displays
- Frequency analysis redesigned as informative graphs (spectral distribution)
- More analytical, less decorative
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

    # Panel sizing - IMAGE IS SMALLER, PANELS ARE LARGER
    PANEL_SIZE_LANDSCAPE = 0.50  # 50% for panels!
    PANEL_SIZE_PORTRAIT = 0.35   # 35%
    PANEL_SIZE_PANORAMA = 0.30   # 30%

    # Padding
    PADDING = 60
    PANEL_PADDING = 25
    SECTION_SPACING = 35

    # Colors - expanded palette
    PRIMARY = (0, 180, 160, 255)       # Teal
    SECONDARY = (220, 80, 40, 255)     # Red-orange
    TERTIARY = (120, 60, 200, 255)     # Purple
    ACCENT = (255, 180, 0, 255)        # Gold
    WARM = (255, 100, 80, 255)         # Warm accent
    COOL = (80, 160, 255, 255)         # Cool accent
    DIM = (70, 70, 70, 255)
    LIGHT_GRAY = (200, 200, 200, 255)
    BACKGROUND = (255, 255, 255, 255)
    PANEL_BG = (245, 247, 250, 255)

    # Typography
    FONT_SIZE_TITLE = 18
    FONT_SIZE_LABEL = 13
    FONT_SIZE_VALUE = 15

    # Overlay intensity - REDUCED since image is smaller
    BASE_IMAGE_FADE = 0.08
    OVERLAY_ALPHA_BOOST = 0.9

    # Analysis parameters
    CONTOUR_LEVELS = 15              # Reduced
    CONTOUR_SMOOTHING = 3.0
    CONTOUR_LINE_WIDTH = 1           # Thinner
    VECTOR_GRID_SPACING = 40         # Sparser
    VECTOR_SCALE = 15
    VECTOR_MIN_THRESHOLD = 0.05
    MAX_FEATURES = 150               # Fewer
    FEATURE_QUALITY = 0.01
    NUM_COLORS = 8                   # More colors!
    GRID_DIVISIONS = 6               # Fewer grid lines


# =============================================================================
# LAYOUT CALCULATION
# =============================================================================

def calculate_layout(img_width, img_height, config=Config):
    """Calculate layout with IMAGE SMALLER, PANELS LARGER."""
    aspect = img_width / img_height

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

    # Calculate layout - IMAGE IS NOW SMALLER
    if layout_mode == LayoutMode.LANDSCAPE:
        # Image gets LESS space, panels get MORE
        img_display_width = int(target_width * 0.85)  # Slightly smaller than input
        img_display_height = int(img_display_width / aspect)

        panel_width = int(img_display_width * config.PANEL_SIZE_LANDSCAPE)
        canvas_width = img_display_width + panel_width + config.PADDING * 3
        canvas_height = max(img_display_height, panel_width) + config.PADDING * 2

        img_x = config.PADDING
        img_y = config.PADDING

        panel_info = {
            'x': img_x + img_display_width + config.PADDING,
            'y': img_y,
            'width': panel_width,
            'height': canvas_height - config.PADDING * 2,
            'orientation': 'vertical'
        }

        target_width = img_display_width
        target_height = img_display_height

    elif layout_mode == LayoutMode.PORTRAIT:
        img_display_height = int(target_height * 0.85)
        img_display_width = int(img_display_height * aspect)

        panel_height = int(img_display_height * config.PANEL_SIZE_PORTRAIT)
        canvas_width = img_display_width + config.PADDING * 2
        canvas_height = img_display_height + panel_height + config.PADDING * 3

        img_x = config.PADDING
        img_y = config.PADDING

        panel_info = {
            'x': img_x,
            'y': img_y + img_display_height + config.PADDING,
            'width': img_display_width,
            'height': panel_height,
            'orientation': 'horizontal'
        }

        target_width = img_display_width
        target_height = img_display_height

    else:  # PANORAMA
        img_display_width = int(target_width * 0.90)
        img_display_height = int(img_display_width / aspect)

        panel_height = int(img_display_height * config.PANEL_SIZE_PANORAMA)
        canvas_width = img_display_width + config.PADDING * 2
        canvas_height = img_display_height + panel_height + config.PADDING * 3

        img_x = config.PADDING
        img_y = config.PADDING + panel_height + config.PADDING

        panel_info = {
            'x': img_x,
            'y': config.PADDING,
            'width': img_display_width,
            'height': panel_height,
            'orientation': 'horizontal'
        }

        target_width = img_display_width
        target_height = img_display_height

    return (canvas_width, canvas_height, img_x, img_y,
            target_width, target_height, panel_info, layout_mode)


# =============================================================================
# CORE ANALYSIS FUNCTIONS (same as V2)
# =============================================================================

def generate_contours(gray, config=Config):
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
    """Compute FFT and return both spectrum and frequency distribution."""
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

    spectrum = (log_magnitude * 255).astype(np.uint8)

    # Compute radial frequency distribution
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Bin by radius
    max_radius = min(cx, cy)
    num_bins = 50
    radial_profile = []

    for i in range(num_bins):
        r_min = (i / num_bins) * max_radius
        r_max = ((i + 1) / num_bins) * max_radius
        mask = (r >= r_min) & (r < r_max)
        if mask.any():
            radial_profile.append(np.mean(log_magnitude[mask]))
        else:
            radial_profile.append(0)

    return spectrum, np.array(radial_profile)


def extract_dominant_colors(image, config=Config):
    pixels = np.array(image).reshape(-1, 3)
    sample_size = min(10000, len(pixels))
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    sample = pixels[indices]

    kmeans = KMeans(n_clusters=config.NUM_COLORS, random_state=42, n_init=10)
    kmeans.fit(sample)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    percentages = counts / counts.sum()

    # Calculate color temperature
    results = []
    for color, pct in zip(colors, percentages):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        # Simple color temperature (red-blue ratio)
        temp = (color[0] - color[2]) / 255.0  # -1 (cool) to +1 (warm)
        results.append((tuple(color), pct, lum, temp))

    results.sort(key=lambda x: -x[1])  # Sort by percentage (most common first)
    return [(r[0], r[1], r[3]) for r in results]


def detect_features(gray, config=Config):
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
    if len(points) < 4:
        return None
    try:
        return Delaunay(points)
    except:
        return None


def draw_constellation(canvas, points, triangulation, offset, scale, config=Config):
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
        draw.ellipse([px-4, py-4, px+4, py+4],
                     fill=(config.TERTIARY[0], config.TERTIARY[1], config.TERTIARY[2], 25))
        draw.ellipse([px-2, py-2, px+2, py+2], fill=config.TERTIARY)
    return canvas


# =============================================================================
# ENHANCED VISUALIZATION PANELS - REDESIGNED FOR ANALYSIS
# =============================================================================

def draw_frequency_analysis(canvas, spectrum, radial_profile, position, size, config=Config):
    """Draw frequency analysis as INFORMATIVE GRAPHS."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Background
    draw.rectangle([x, y, x + w, y + h], fill=config.PANEL_BG, outline=config.PRIMARY, width=2)

    # Title
    draw.text((x + 15, y + 12), "SPECTRAL ENERGY DISTRIBUTION", fill=config.PRIMARY, font=None)
    draw.text((x + 15, y + 30), "Frequency domain analysis (2D-FFT)", fill=config.DIM, font=None)

    # Graph area
    graph_y = y + 60
    graph_h = h - 120
    graph_x = x + 40
    graph_w = w - 60

    # Draw axes
    draw.line([(graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h)],
             fill=config.DIM, width=2)  # X axis
    draw.line([(graph_x, graph_y), (graph_x, graph_y + graph_h)],
             fill=config.DIM, width=2)  # Y axis

    # Plot radial frequency distribution as area chart
    if len(radial_profile) > 0:
        max_val = radial_profile.max()
        if max_val > 0:
            points = []
            for i, val in enumerate(radial_profile):
                px = graph_x + int((i / len(radial_profile)) * graph_w)
                py = graph_y + graph_h - int((val / max_val) * graph_h)
                points.append((px, py))

            # Draw filled area
            if len(points) > 1:
                # Create polygon for filled area
                area_points = [(graph_x, graph_y + graph_h)] + points + [(graph_x + graph_w, graph_y + graph_h)]
                draw.polygon(area_points, fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], 60))

                # Draw line on top
                draw.line(points, fill=config.PRIMARY, width=2)

                # Highlight peaks
                for i in range(1, len(radial_profile) - 1):
                    if radial_profile[i] > radial_profile[i-1] and radial_profile[i] > radial_profile[i+1]:
                        if radial_profile[i] > max_val * 0.6:  # Significant peaks only
                            px = graph_x + int((i / len(radial_profile)) * graph_w)
                            py = graph_y + graph_h - int((radial_profile[i] / max_val) * graph_h)
                            draw.ellipse([px-4, py-4, px+4, py+4], fill=config.SECONDARY)

    # Labels
    draw.text((graph_x - 5, graph_y - 20), "Energy", fill=config.DIM, font=None)
    draw.text((graph_x, graph_y + graph_h + 10), "Low", fill=config.DIM, font=None)
    draw.text((graph_x + graph_w - 30, graph_y + graph_h + 10), "High", fill=config.DIM, font=None)
    draw.text((graph_x + graph_w // 2 - 40, graph_y + graph_h + 10), "Frequency →", fill=config.DIM, font=None)

    # Statistics
    low_freq_energy = np.mean(radial_profile[:len(radial_profile)//4])
    high_freq_energy = np.mean(radial_profile[3*len(radial_profile)//4:])

    stats_y = y + h - 45
    draw.text((x + 15, stats_y), f"Low freq energy: {low_freq_energy:.2f}", fill=config.DIM, font=None)
    draw.text((x + 15, stats_y + 18), f"High freq energy: {high_freq_energy:.2f}", fill=config.DIM, font=None)

    return canvas


def draw_color_analysis_large(canvas, colors, position, size, config=Config):
    """LARGE color analysis - this is now the star of the show."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Background
    draw.rectangle([x, y, x + w, y + h], fill=config.PANEL_BG, outline=config.PRIMARY, width=2)

    # Title with more emphasis
    draw.text((x + 15, y + 12), "CHROMATIC COMPOSITION", fill=config.PRIMARY, font=None)
    total_pct = sum(c[1] for c in colors)
    draw.text((x + 15, y + 30), f"{len(colors)} dominant colors · 100% coverage", fill=config.DIM, font=None)

    # Color bars - MUCH LARGER
    bar_start_y = y + 60
    bar_height = max(35, (h - 80) // len(colors))  # Minimum 35px per bar
    max_bar_width = w - 180

    for i, (color, pct, temp) in enumerate(colors):
        by = bar_start_y + i * bar_height
        bar_width = int(max_bar_width * pct)

        # Background bar (full width, light)
        draw.rectangle([x + 15, by, x + 15 + max_bar_width, by + bar_height - 10],
                      fill=(230, 230, 230, 255), outline=config.LIGHT_GRAY, width=1)

        # Actual color bar with gradient effect
        for offset in range(3, 0, -1):
            alpha = 255 - offset * 30
            draw.rectangle([x + 15 + offset, by + offset, x + 15 + bar_width - offset, by + bar_height - 10 - offset],
                          fill=(color[0], color[1], color[2], alpha))

        # Main color bar
        draw.rectangle([x + 15, by, x + 15 + bar_width, by + bar_height - 10],
                      fill=(color[0], color[1], color[2], 255),
                      outline=(color[0]//3, color[1]//3, color[2]//3, 255), width=2)

        # LARGE percentage text
        pct_text = f"{pct*100:.1f}%"
        text_x = x + 15 + max_bar_width + 15
        draw.text((text_x, by + 3), pct_text, fill=config.PRIMARY, font=None)

        # Hex code
        hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        draw.text((text_x, by + 20), hex_code, fill=config.DIM, font=None)

        # Color temperature indicator
        if temp > 0.2:
            temp_label = "warm"
            temp_color = config.WARM
        elif temp < -0.2:
            temp_label = "cool"
            temp_color = config.COOL
        else:
            temp_label = "neutral"
            temp_color = config.DIM

        # Small temperature dot
        dot_x = x + 15 + bar_width + 5
        if dot_x < x + 15 + max_bar_width:
            draw.ellipse([dot_x, by + bar_height//2 - 3, dot_x + 6, by + bar_height//2 + 3],
                        fill=temp_color)

    return canvas


def draw_stats_panel(canvas, stats, position, size, config=Config):
    """Draw statistics panel."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Background
    draw.rectangle([x, y, x + w, y + h], fill=config.PANEL_BG, outline=config.PRIMARY, width=2)

    # Title
    draw.text((x + 15, y + 12), "ANALYSIS SUMMARY", fill=config.PRIMARY, font=None)

    # Stats
    stat_y = y + 45
    for label, value in stats.items():
        # Label
        draw.text((x + 20, stat_y), label, fill=config.DIM, font=None)
        # Value (right-aligned)
        value_text = str(value)
        draw.text((x + w - 65, stat_y), value_text, fill=config.SECONDARY, font=None)
        stat_y += 28

    return canvas


def draw_analysis_panels(canvas, spectrum, radial_profile, colors, panel_info, stats, config=Config):
    """Draw all panels with ANALYSIS FOCUS."""
    px, py = panel_info['x'], panel_info['y']
    pw, ph = panel_info['width'], panel_info['height']

    if panel_info['orientation'] == 'vertical':
        # Vertical layout - COLOR IS NOW THE BIGGEST
        spacing = config.SECTION_SPACING

        # COLOR ANALYSIS - LARGEST (50% of panel)
        color_h = int(ph * 0.50)
        draw_color_analysis_large(canvas, colors, (px, py), (pw, color_h), config)

        # FREQUENCY ANALYSIS - Medium (35% of panel)
        freq_y = py + color_h + spacing
        freq_h = int(ph * 0.35)
        draw_frequency_analysis(canvas, spectrum, radial_profile, (px, freq_y), (pw, freq_h), config)

        # STATS - Small (remaining space)
        stats_y = freq_y + freq_h + spacing
        stats_h = ph - (stats_y - py) - 10
        if stats_h > 100:
            draw_stats_panel(canvas, stats, (px, stats_y), (pw, stats_h), config)

    else:
        # Horizontal layout
        section_w = pw // 3
        spacing = 20

        # Color (left - largest)
        color_w = int(section_w * 1.5) - spacing
        color_h = ph - 20
        draw_color_analysis_large(canvas, colors, (px, py), (color_w, color_h), config)

        # Frequency (middle)
        freq_x = px + int(section_w * 1.5)
        freq_w = section_w - spacing
        freq_h = ph - 20
        draw_frequency_analysis(canvas, spectrum, radial_profile, (freq_x, py), (freq_w, freq_h), config)

        # Stats (right)
        stats_x = px + int(section_w * 2.5)
        stats_w = pw - (stats_x - px) - 10
        stats_h = ph - 20
        draw_stats_panel(canvas, stats, (stats_x, py), (stats_w, stats_h), config)


def draw_hud_frame(canvas, image_bounds, original_size, layout_mode, config=Config):
    """Simplified HUD - image is reference, not focus."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y, w, h = image_bounds
    orig_w, orig_h = original_size

    # Minimal grid
    grid_lines = 8
    for i in range(1, grid_lines):
        lx = x + (w * i // grid_lines)
        alpha = 20 if i % 2 != 0 else 35
        draw.line([(lx, y), (lx, y + h)],
                 fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)

        ly = y + (h * i // grid_lines)
        draw.line([(x, ly), (x + w, ly)],
                 fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)

    # Simple corner brackets
    bracket_len = 30
    draw.line([(x, y), (x + bracket_len, y)], fill=config.PRIMARY, width=2)
    draw.line([(x, y), (x, y + bracket_len)], fill=config.PRIMARY, width=2)
    draw.line([(x + w, y), (x + w - bracket_len, y)], fill=config.PRIMARY, width=2)
    draw.line([(x + w, y), (x + w, y + bracket_len)], fill=config.PRIMARY, width=2)
    draw.line([(x, y + h), (x + bracket_len, y + h)], fill=config.PRIMARY, width=2)
    draw.line([(x, y + h), (x, y + h - bracket_len)], fill=config.PRIMARY, width=2)
    draw.line([(x + w, y + h), (x + w - bracket_len, y + h)], fill=config.PRIMARY, width=2)
    draw.line([(x + w, y + h), (x + w, y + h - bracket_len)], fill=config.PRIMARY, width=2)

    # Header
    canvas_w, canvas_h = canvas.size
    draw.text((15, 15), f"ORBITAL SURVEY v3", fill=config.PRIMARY, font=None)
    draw.text((15, 33), f"Reference: {orig_w}×{orig_h}px", fill=config.DIM, font=None)

    draw.text((canvas_w - 200, 15), "ANALYSIS COMPLETE", fill=config.SECONDARY, font=None)

    # Simple crosshair
    cx, cy = x + w // 2, y + h // 2
    cross_size = 20
    draw.line([(cx - cross_size, cy), (cx - 6, cy)], fill=config.PRIMARY, width=2)
    draw.line([(cx + 6, cy), (cx + cross_size, cy)], fill=config.PRIMARY, width=2)
    draw.line([(cx, cy - cross_size), (cx, cy - 6)], fill=config.PRIMARY, width=2)
    draw.line([(cx, cy + 6), (cx, cy + cross_size)], fill=config.PRIMARY, width=2)

    return canvas


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_image(input_path, output_path=None, config=Config):
    """Process image with ANALYSIS FOCUS."""

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading: {input_path}")

    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")

    orig_w, orig_h = img.size
    print(f"  Original: {orig_w}×{orig_h}")

    # Calculate layout
    canvas_w, canvas_h, img_x, img_y, img_w, img_h, panel_info, layout_mode = \
        calculate_layout(orig_w, orig_h, config)

    print(f"  Layout: {layout_mode.value}")
    print(f"  Canvas: {canvas_w}×{canvas_h}")
    print(f"  Image: {img_w}×{img_h} (reference)")
    print(f"  Panel: {panel_info['width']}×{panel_info['height']} (primary focus)")

    # Create canvas
    canvas = Image.new('RGBA', (canvas_w, canvas_h), config.BACKGROUND)

    # Resize image
    img_resized = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply base image (minimal fade)
    base_img = Image.fromarray(img_array)
    if config.BASE_IMAGE_FADE > 0:
        base_img = Image.blend(base_img,
                               Image.new('RGB', base_img.size, (255, 255, 255)),
                               alpha=config.BASE_IMAGE_FADE)
    canvas.paste(base_img, (img_x, img_y))

    scale = 1.0
    offset = (img_x, img_y)

    # Generate layers (reduced intensity)
    print("  [1/6] Contours...")
    contours = generate_contours(gray, config)
    canvas = draw_contours(canvas, contours, offset, scale, config)

    print("  [2/6] Gradients...")
    vectors, max_mag = compute_gradient_field(gray, config)
    canvas = draw_vector_field(canvas, vectors, max_mag, offset, scale, config)

    print("  [3/6] Frequency analysis...")
    spectrum, radial_profile = compute_fft_spectrum(gray)

    print("  [4/6] Color analysis...")
    colors = extract_dominant_colors(img_resized, config)

    print("  [5/6] Features...")
    features = detect_features(gray, config)
    triangulation = compute_delaunay(features)
    canvas = draw_constellation(canvas, features, triangulation, offset, scale, config)

    print("  [6/6] HUD and panels...")
    image_bounds = (img_x, img_y, img_w, img_h)
    canvas = draw_hud_frame(canvas, image_bounds, (orig_w, orig_h), layout_mode, config)

    # Draw analysis panels - THESE ARE THE FOCUS
    stats = {
        "Contours": config.CONTOUR_LEVELS,
        "Features": len(features),
        "Vectors": len(vectors),
        "Colors": config.NUM_COLORS,
    }
    draw_analysis_panels(canvas, spectrum, radial_profile, colors, panel_info, stats, config)

    # Legend
    draw = ImageDraw.Draw(canvas, 'RGBA')
    legend_y = canvas_h - 30
    legend_items = [
        (config.PRIMARY, "CONTOURS"),
        (config.SECONDARY, "GRADIENTS"),
        (config.TERTIARY, "FEATURES"),
    ]
    lx = config.PADDING
    for color, label in legend_items:
        draw.rectangle([lx, legend_y, lx + 14, legend_y + 14], fill=color, outline=config.DIM, width=1)
        draw.text((lx + 20, legend_y + 1), label, fill=config.DIM, font=None)
        lx += 120

    # Save
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_survey_v3{input_file.suffix}"

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
            print("No file selected.")
            sys.exit(0)
        output_path = None

    try:
        result = process_image(input_path, output_path)
        print(f"\n✓ Complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
