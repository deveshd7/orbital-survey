"""
ORBITAL SURVEY PROTOCOL - V4.1
===============================
Clean 50/50 split layout:
  LEFT:  Source image with contour/gradient/constellation overlays + HUD
  RIGHT: Two stacked analysis panels
    TOP:    Color distribution (proportional bars + hex + percentages)
    BOTTOM: Color Spectrograph — frequency distribution across hue spectrum
            showing which colors appear most in the image
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from pathlib import Path
import sys
import colorsys


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    OUTPUT_WIDTH = 3840
    PANEL_SPLIT = 0.50
    PADDING = 60
    PANEL_GAP = 40

    PRIMARY = (0, 180, 160, 255)
    SECONDARY = (220, 80, 40, 255)
    TERTIARY = (120, 60, 200, 255)
    DIM = (80, 80, 80, 255)
    LIGHT_DIM = (140, 140, 140, 255)
    BACKGROUND = (255, 255, 255, 255)
    PANEL_BG = (248, 249, 252, 255)
    PANEL_BORDER = (0, 180, 160, 180)

    BASE_IMAGE_FADE = 0.10
    CONTOUR_LEVELS = 18
    CONTOUR_SMOOTHING = 3.0
    CONTOUR_LINE_WIDTH = 2
    VECTOR_GRID_SPACING = 35
    VECTOR_SCALE = 18
    VECTOR_MIN_THRESHOLD = 0.04
    MAX_FEATURES = 180
    FEATURE_QUALITY = 0.01
    NUM_COLORS = 10
    GRID_DIVISIONS = 8



# =============================================================================
# OVERLAY LAYERS
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


def draw_contours(canvas, contours_all, offset, config=Config):
    draw = ImageDraw.Draw(canvas, 'RGBA')
    for level, contours in contours_all:
        alpha = min(int(80 + (level / 255) * 120), 200)
        color = (*config.PRIMARY[:3], alpha)
        for contour in contours:
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue
            scaled = [(int(p[0] + offset[0]), int(p[1] + offset[1])) for p in points]
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


def draw_vector_field(canvas, vectors, max_mag, offset, config=Config):
    draw = ImageDraw.Draw(canvas, 'RGBA')
    for x, y, mag, ang in vectors:
        if mag < max_mag * config.VECTOR_MIN_THRESHOLD:
            continue
        norm_mag = max((mag / max_mag) * config.VECTOR_SCALE, 3)
        sx = int(x + offset[0])
        sy = int(y + offset[1])
        ex = int(sx + np.cos(ang) * norm_mag)
        ey = int(sy + np.sin(ang) * norm_mag)
        alpha = min(int(50 + (mag / max_mag) * 160), 220)
        color = (*config.SECONDARY[:3], alpha)
        draw.line([(sx, sy), (ex, ey)], fill=color, width=1)
        draw.ellipse([sx - 1, sy - 1, sx + 1, sy + 1], fill=color)
    return canvas


def detect_features(gray, config=Config):
    h, w = gray.shape
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    threshold = config.FEATURE_QUALITY * harris.max()
    grid_size = int(np.sqrt(config.MAX_FEATURES))
    cell_h, cell_w = h // grid_size, w // grid_size
    points = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y_start, y_end = gy * cell_h, min((gy + 1) * cell_h, h)
            x_start, x_end = gx * cell_w, min((gx + 1) * cell_w, w)
            cell = harris[y_start:y_end, x_start:x_end]
            if cell.max() > threshold * 0.5:
                loc = np.unravel_index(cell.argmax(), cell.shape)
                points.append([x_start + loc[1], y_start + loc[0]])
            else:
                cx = np.clip(x_start + cell_w // 2 + np.random.randint(-cell_w // 4, cell_w // 4),
                             x_start, x_end - 1)
                cy = np.clip(y_start + cell_h // 2 + np.random.randint(-cell_h // 4, cell_h // 4),
                             y_start, y_end - 1)
                points.append([cx, cy])
    return np.array(points) if points else np.array([]).reshape(0, 2)


def draw_constellation(canvas, points, offset, config=Config):
    draw = ImageDraw.Draw(canvas, 'RGBA')
    if len(points) == 0:
        return canvas
    scaled = [(int(p[0] + offset[0]), int(p[1] + offset[1])) for p in points]
    if len(points) >= 4:
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edges.add(tuple(sorted([simplex[i], simplex[(i + 1) % 3]])))
            for i, j in edges:
                if i >= len(scaled) or j >= len(scaled):
                    continue
                p1, p2 = scaled[i], scaled[j]
                dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                alpha = max(15, min(80, int(180 - dist * 0.35)))
                draw.line([p1, p2], fill=(*config.TERTIARY[:3], alpha), width=1)
        except Exception:
            pass
    for px, py in scaled:
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(*config.TERTIARY[:3], 20))
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=config.TERTIARY)
    return canvas


def draw_hud(canvas, img_bounds, orig_size, config=Config):
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y, w, h = img_bounds
    orig_w, orig_h = orig_size

    for i in range(1, config.GRID_DIVISIONS):
        lx = x + w * i // config.GRID_DIVISIONS
        ly = y + h * i // config.GRID_DIVISIONS
        a = 25 if i % 2 else 45
        draw.line([(lx, y), (lx, y + h)], fill=(*config.PRIMARY[:3], a), width=1)
        draw.line([(x, ly), (x + w, ly)], fill=(*config.PRIMARY[:3], a), width=1)

    bl = 35
    for cx, cy, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        draw.line([(cx, cy), (cx + dx * bl, cy)], fill=config.PRIMARY, width=2)
        draw.line([(cx, cy), (cx, cy + dy * bl)], fill=config.PRIMARY, width=2)

    cx, cy = x + w // 2, y + h // 2
    cs = 22
    for s, e in [((cx - cs, cy), (cx - 6, cy)), ((cx + 6, cy), (cx + cs, cy)),
                 ((cx, cy - cs), (cx, cy - 6)), ((cx, cy + 6), (cx, cy + cs))]:
        draw.line([s, e], fill=(*config.PRIMARY[:3], 100), width=1)

    cw = canvas.size[0]
    draw.text((15, 12), "ORBITAL SURVEY v4", fill=config.PRIMARY)
    draw.text((15, 30), f"Reference: {orig_w}\u00d7{orig_h}px", fill=config.DIM)
    draw.text((cw - 200, 12), "SCAN COMPLETE", fill=config.SECONDARY)

    legend_y = canvas.size[1] - 30
    lx = config.PADDING
    for color, label in [(config.PRIMARY, "CONTOURS"), (config.SECONDARY, "GRADIENT"), (config.TERTIARY, "FEATURES")]:
        draw.rectangle([lx, legend_y, lx + 12, legend_y + 12], fill=color, outline=config.DIM, width=1)
        draw.text((lx + 18, legend_y - 1), label, fill=config.DIM)
        lx += 120
    return canvas


# =============================================================================
# RIGHT PANEL TOP: COLOR DISTRIBUTION
# =============================================================================

def extract_colors(image, config=Config):
    pixels = np.array(image).reshape(-1, 3)
    sample = pixels[np.random.choice(len(pixels), min(15000, len(pixels)), replace=False)]
    kmeans = KMeans(n_clusters=config.NUM_COLORS, random_state=42, n_init=10)
    kmeans.fit(sample)
    colors = kmeans.cluster_centers_.astype(int)
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    pcts = counts / counts.sum()
    results = []
    for color, pct in zip(colors, pcts):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        h_val, s_val, v_val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        results.append(((r, g, b), float(pct), h_val, s_val, v_val))
    results.sort(key=lambda x: -x[1])
    return results


def draw_color_panel(canvas, colors, position, size, config=Config):
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    draw.rectangle([x, y, x + w, y + h], fill=config.PANEL_BG, outline=config.PANEL_BORDER, width=2)

    title_h = 55
    draw.text((x + 20, y + 14), "CHROMATIC DISTRIBUTION", fill=config.PRIMARY)
    draw.text((x + 20, y + 32), f"{len(colors)} dominant clusters \u00b7 K-means extraction", fill=config.LIGHT_DIM)

    bar_area_y = y + title_h
    bar_area_h = h - title_h - 15
    bar_h = max(28, bar_area_h // len(colors))
    max_bar_w = w - 220

    for i, (rgb, pct, h_val, s_val, v_val) in enumerate(colors):
        by = bar_area_y + i * bar_h
        bw = int(max_bar_w * (pct / colors[0][1]))
        bw = max(bw, 20)

        r, g, b = rgb
        draw.rectangle([x + 18, by + 4, x + 18 + max_bar_w, by + bar_h - 6],
                       fill=(235, 235, 235, 255))
        draw.rectangle([x + 19, by + 5, x + 18 + bw + 1, by + bar_h - 5],
                       fill=(r // 2, g // 2, b // 2, 80))
        draw.rectangle([x + 18, by + 4, x + 18 + bw, by + bar_h - 6],
                       fill=(r, g, b, 255),
                       outline=(max(r - 40, 0), max(g - 40, 0), max(b - 40, 0), 200), width=1)

        tx = x + 18 + max_bar_w + 12
        draw.text((tx, by + 2), f"{pct * 100:.1f}%", fill=config.PRIMARY)
        draw.text((tx, by + bar_h // 2), f"#{r:02X}{g:02X}{b:02X}", fill=config.DIM)
    return canvas


# =============================================================================
# RIGHT PANEL BOTTOM: RADIAL CHROMATOGRAPH — clean & well-designed
# =============================================================================

def compute_color_spectrograph(image, num_bins=180):
    """
    Analyze image and create hue frequency distribution.
    Returns array of frequencies for each hue bin (0-360 degrees).
    """
    arr = np.array(image)
    h, w, _ = arr.shape

    # Convert to HSV and extract hues
    hues = []
    saturations = []

    for i in range(0, h, 2):  # Sample every other row for speed
        for j in range(0, w, 2):
            r, g, b = arr[i, j] / 255.0
            h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)

            # Only count pixels with sufficient saturation and value
            if s_val > 0.1 and v_val > 0.1:
                hues.append(h_val * 360)
                saturations.append(s_val)

    # Create histogram bins
    hist, bin_edges = np.histogram(hues, bins=num_bins, range=(0, 360))

    # Smooth the histogram
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

    return hist_smooth, bin_edges


def draw_color_spectrograph(canvas, spectrum_data, bin_edges, position, size, config=Config):
    """
    Draw color spectrograph - frequency distribution across hue spectrum.
    Like a rainbow bar where height shows how much of each color appears.
    """
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Panel background
    draw.rectangle([x, y, x + w, y + h], fill=config.PANEL_BG, outline=config.PANEL_BORDER, width=2)

    # Title
    title_h = 55
    draw.text((x + 20, y + 14), "COLOR SPECTROGRAPH", fill=config.PRIMARY)
    draw.text((x + 20, y + 32), "Hue frequency distribution · which colors dominate", fill=config.LIGHT_DIM)

    # Spec graph area
    graph_margin_x = 40
    graph_margin_y = 30
    graph_x = x + graph_margin_x
    graph_y = y + title_h + graph_margin_y
    graph_w = w - graph_margin_x * 2
    graph_h = h - title_h - graph_margin_y - 60

    # Background
    draw.rectangle([graph_x, graph_y, graph_x + graph_w, graph_y + graph_h],
                   fill=(255, 255, 255, 255), outline=(200, 200, 200, 255), width=1)

    # Grid lines
    for i in range(1, 5):
        gy = graph_y + (graph_h * i // 5)
        draw.line([(graph_x, gy), (graph_x + graph_w, gy)],
                 fill=(240, 240, 240, 255), width=1)

    # Normalize spectrum data
    max_val = spectrum_data.max()
    if max_val > 0:
        normalized = spectrum_data / max_val
    else:
        normalized = spectrum_data

    # Draw spectrum bars
    num_bins = len(spectrum_data)
    bin_width = graph_w / num_bins

    for i, intensity in enumerate(normalized):
        # Calculate hue for this bin
        hue = (i / num_bins) * 360

        # Convert hue to RGB for bar color
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 0.9, 0.95)]

        # Bar dimensions
        bx = graph_x + i * bin_width
        bar_height = intensity * graph_h
        by = graph_y + graph_h - bar_height

        # Draw bar with gradient effect
        if bar_height > 2:
            # Main bar
            draw.rectangle([bx, by, bx + bin_width, graph_y + graph_h],
                          fill=(r, g, b, 200), outline=None)

            # Lighter top for depth
            if bar_height > 10:
                light_r = min(255, int(r * 1.3))
                light_g = min(255, int(g * 1.3))
                light_b = min(255, int(b * 1.3))
                draw.rectangle([bx, by, bx + bin_width, by + 5],
                              fill=(light_r, light_g, light_b, 180), outline=None)

    # Draw spectrum reference bar at bottom
    rainbow_y = graph_y + graph_h + 10
    rainbow_h = 20

    for i in range(graph_w):
        hue = (i / graph_w) * 360
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)]
        draw.line([(graph_x + i, rainbow_y), (graph_x + i, rainbow_y + rainbow_h)],
                 fill=(r, g, b, 255))

    draw.rectangle([graph_x, rainbow_y, graph_x + graph_w, rainbow_y + rainbow_h],
                  outline=(180, 180, 180, 255), width=1)

    # Axis labels
    draw.text((graph_x, rainbow_y + rainbow_h + 4), "0°", fill=config.DIM)
    draw.text((graph_x + graph_w // 4 - 10, rainbow_y + rainbow_h + 4), "90°", fill=config.LIGHT_DIM)
    draw.text((graph_x + graph_w // 2 - 15, rainbow_y + rainbow_h + 4), "180°", fill=config.LIGHT_DIM)
    draw.text((graph_x + 3 * graph_w // 4 - 15, rainbow_y + rainbow_h + 4), "270°", fill=config.LIGHT_DIM)
    draw.text((graph_x + graph_w - 25, rainbow_y + rainbow_h + 4), "360°", fill=config.DIM)

    # Y-axis label
    draw.text((x + 8, graph_y - 20), "Frequency", fill=config.DIM)
    draw.text((graph_x + graph_w // 2 - 20, rainbow_y + rainbow_h + 22), "Hue (degrees)", fill=config.DIM)

    # Peak detection - mark dominant hues
    peaks = []
    for i in range(2, len(normalized) - 2):
        if normalized[i] > normalized[i-1] and normalized[i] > normalized[i+1]:
            if normalized[i] > 0.5:  # Significant peaks only
                peaks.append((i, normalized[i]))

    # Mark peaks
    for peak_idx, peak_val in peaks[:3]:  # Top 3 peaks
        bx = graph_x + peak_idx * bin_width
        by = graph_y + graph_h - peak_val * graph_h

        # Peak marker
        draw.ellipse([bx - 4, by - 4, bx + 4, by + 4],
                    fill=config.SECONDARY, outline=(255, 255, 255, 255), width=2)

        # Peak label
        hue_deg = int((peak_idx / num_bins) * 360)
        draw.text((bx - 15, by - 20), f"{hue_deg}°", fill=config.SECONDARY)

    return canvas


# =============================================================================
# MAIN COMPOSITOR
# =============================================================================

def process_image(input_path, output_path=None, config=Config):
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    print(f"Loading: {input_path}")
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h
    print(f"  Source: {orig_w}\u00d7{orig_h}  aspect={aspect:.2f}")

    canvas_w = config.OUTPUT_WIDTH
    img_half_w = int((canvas_w - config.PADDING * 3) * config.PANEL_SPLIT)
    panel_half_w = canvas_w - img_half_w - config.PADDING * 3

    img_display_h = int(img_half_w / aspect)
    canvas_h = img_display_h + config.PADDING * 2

    min_panel_h = 800
    if canvas_h < min_panel_h + config.PADDING * 2:
        canvas_h = min_panel_h + config.PADDING * 2
        img_display_h = canvas_h - config.PADDING * 2

    img_x = config.PADDING
    img_y = config.PADDING

    panel_x = img_x + img_half_w + config.PADDING
    panel_y = img_y
    panel_w = panel_half_w
    panel_total_h = img_display_h

    print(f"  Canvas: {canvas_w}\u00d7{canvas_h}")
    print(f"  Image area: {img_half_w}\u00d7{img_display_h}")
    print(f"  Panel area: {panel_w}\u00d7{panel_total_h}")

    canvas = Image.new('RGBA', (canvas_w, canvas_h), config.BACKGROUND)

    img_resized = img.resize((img_half_w, img_display_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    base = Image.fromarray(img_array)
    if config.BASE_IMAGE_FADE > 0:
        base = Image.blend(base, Image.new('RGB', base.size, (255, 255, 255)),
                           alpha=config.BASE_IMAGE_FADE)
    canvas.paste(base, (img_x, img_y))
    offset = (img_x, img_y)

    print("  [1/5] Contours...")
    contours = generate_contours(gray, config)
    canvas = draw_contours(canvas, contours, offset, config)

    print("  [2/5] Gradient field...")
    vectors, max_mag = compute_gradient_field(gray, config)
    canvas = draw_vector_field(canvas, vectors, max_mag, offset, config)

    print("  [3/5] Feature constellation...")
    features = detect_features(gray, config)
    canvas = draw_constellation(canvas, features, offset, config)

    print("  [4/5] Color extraction...")
    colors = extract_colors(img_resized, config)

    print("  [5/5] Color spectrograph...")
    

    img_bounds = (img_x, img_y, img_half_w, img_display_h)
    canvas = draw_hud(canvas, img_bounds, (orig_w, orig_h), config)

    color_panel_h = int(panel_total_h * 0.45)
    draw_color_panel(canvas, colors, (panel_x, panel_y), (panel_w, color_panel_h), config)

    radial_panel_y = panel_y + color_panel_h + config.PANEL_GAP
    radial_panel_h = panel_total_h - color_panel_h - config.PANEL_GAP
    spectrum_data, bin_edges = compute_color_spectrograph(img_resized)
    draw_color_spectrograph(canvas, spectrum_data, bin_edges, (panel_x, radial_panel_y), (panel_w, radial_panel_h), config)

    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_survey_v4.png"

    final = Image.new('RGB', canvas.size, (255, 255, 255))
    final.paste(canvas, mask=canvas.split()[3])
    final.save(str(output_path), quality=95)

    print(f"\n\u2713 Saved: {output_path}")
    return str(output_path)


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
        path = filedialog.askopenfilename(
            title="Select Image for Orbital Survey",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")]
        )
        root.destroy()
        return path

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
        print(f"\u2713 Complete!")
    except Exception as e:
        print(f"\u2717 Error: {e}")
        sys.exit(1)
