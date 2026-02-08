"""
ORBITAL SURVEY PROTOCOL - V4.1
===============================
Clean 50/50 split layout:
  LEFT:  Source image with contour/gradient/constellation overlays + HUD
  RIGHT: Two stacked analysis panels
    TOP:    Color distribution (proportional bars + hex + percentages)
    BOTTOM: Radial Chromatograph — clean circular color wheel
            showing color distribution with elegant design
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

def draw_radial_chromatograph(canvas, dominant_colors, position, size, config=Config):
    """
    Radial chromatograph - clean circular color wheel showing color distribution.
    Features concentric rings for saturation/value and radial segments for each color.
    """
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size

    # Panel background
    draw.rectangle([x, y, x + w, y + h], fill=config.PANEL_BG, outline=config.PANEL_BORDER, width=2)

    # Title
    title_h = 55
    draw.text((x + 20, y + 14), "RADIAL CHROMATOGRAPH", fill=config.PRIMARY)
    draw.text((x + 20, y + 32), f"{len(dominant_colors)} color clusters · radial distribution", fill=config.LIGHT_DIM)

    # Calculate center and radius
    available_h = h - title_h - 90
    available_w = w - 100
    radius = min(available_w, available_h) // 2 - 20

    center_x = x + w // 2
    center_y = y + title_h + available_h // 2 + 20

    # Draw background circle
    draw.ellipse([center_x - radius - 5, center_y - radius - 5,
                  center_x + radius + 5, center_y + radius + 5],
                 fill=(255, 255, 255, 255), outline=(220, 220, 220, 255), width=2)

    # Draw concentric rings for reference
    num_rings = 4
    for i in range(1, num_rings + 1):
        r = int(radius * (i / num_rings))
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=(240, 240, 240, 255), width=1)

    # Draw radial segments for each color
    total_angle = 360
    start_angle = -90

    for i, (rgb, pct, h_val, s_val, v_val) in enumerate(dominant_colors):
        angle_size = pct * total_angle
        segment_radius = int(radius * (0.4 + s_val * 0.6))
        r, g, b = rgb

        # Create pie slice points
        num_points = max(3, int(angle_size / 2))
        points = [(center_x, center_y)]
        for j in range(num_points + 1):
            angle_rad = np.radians(start_angle + (j / num_points) * angle_size)
            px = int(center_x + segment_radius * np.cos(angle_rad))
            py = int(center_y + segment_radius * np.sin(angle_rad))
            points.append((px, py))

        # Draw filled segment
        draw.polygon(points, fill=(r, g, b, 255), outline=(r//2, g//2, b//2, 255), width=2)

        # Inner gradient for depth
        inner_radius = int(segment_radius * 0.7)
        inner_points = [(center_x, center_y)]
        for j in range(num_points + 1):
            angle_rad = np.radians(start_angle + (j / num_points) * angle_size)
            px = int(center_x + inner_radius * np.cos(angle_rad))
            py = int(center_y + inner_radius * np.sin(angle_rad))
            inner_points.append((px, py))

        light_r = min(255, int(r * 1.2))
        light_g = min(255, int(g * 1.2))
        light_b = min(255, int(b * 1.2))
        draw.polygon(inner_points, fill=(light_r, light_g, light_b, 180))

        # Percentage label
        mid_angle = start_angle + angle_size / 2
        label_radius = segment_radius * 0.85
        label_x = int(center_x + label_radius * np.cos(np.radians(mid_angle)))
        label_y = int(center_y + label_radius * np.sin(np.radians(mid_angle)))

        pct_text = f"{pct * 100:.1f}%"
        text_bbox = draw.textbbox((label_x, label_y), pct_text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        draw.rectangle([label_x - text_w//2 - 4, label_y - text_h//2 - 2,
                       label_x + text_w//2 + 4, label_y + text_h//2 + 2],
                      fill=(255, 255, 255, 220), outline=None)

        draw.text((label_x - text_w//2, label_y - text_h//2), pct_text, fill=config.PRIMARY)

        start_angle += angle_size

    # Center circle
    center_radius = 35
    draw.ellipse([center_x - center_radius, center_y - center_radius,
                  center_x + center_radius, center_y + center_radius],
                 fill=(250, 250, 250, 255), outline=config.PRIMARY, width=3)

    draw.text((center_x - 25, center_y - 18), f"{len(dominant_colors)}", fill=config.PRIMARY)
    draw.text((center_x - 22, center_y + 2), "colors", fill=config.DIM)

    # Legend
    legend_y = y + h - 35
    legend_text = "Radius: Saturation · Segments: Distribution"
    text_bbox = draw.textbbox((0, 0), legend_text)
    text_w = text_bbox[2] - text_bbox[0]
    legend_x = center_x - text_w // 2
    draw.text((legend_x, legend_y), legend_text, fill=config.LIGHT_DIM)

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

    print("  [5/5] Radial chromatograph...")
    

    img_bounds = (img_x, img_y, img_half_w, img_display_h)
    canvas = draw_hud(canvas, img_bounds, (orig_w, orig_h), config)

    color_panel_h = int(panel_total_h * 0.45)
    draw_color_panel(canvas, colors, (panel_x, panel_y), (panel_w, color_panel_h), config)

    radial_panel_y = panel_y + color_panel_h + config.PANEL_GAP
    radial_panel_h = panel_total_h - color_panel_h - config.PANEL_GAP
    draw_radial_chromatograph(canvas, colors, (panel_x, radial_panel_y), (panel_w, radial_panel_h), config)

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
