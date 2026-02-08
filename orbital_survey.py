"""
ORBITAL SURVEY PROTOCOL
=======================
Transforms landscape photographs into sci-fi planetary survey visualizations.

Layers:
  1. Luminance Contours - Topographic lines from brightness
  2. Gradient Vector Field - Intensity change direction arrows
  3. Frequency Spectrum Inset - 2D FFT signature
  4. Dominant Color Spectra - Emission-line style palette
  5. Feature Constellation - Corner detection + Delaunay mesh
  6. HUD Frame - Coordinate overlays and technical readouts
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from pathlib import Path
import colorsys


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Output
    OUTPUT_WIDTH = 1920
    PADDING = 80
    
    # Colors (RGBA) - adjusted for white background
    PRIMARY = (0, 180, 160, 255)       # Teal
    SECONDARY = (220, 80, 40, 255)     # Red-orange
    TERTIARY = (120, 60, 200, 255)     # Purple
    DIM = (100, 100, 100, 200)         # Gray for text
    BACKGROUND = (255, 255, 255, 255)  # White
    
    # Contours
    CONTOUR_LEVELS = 20
    CONTOUR_SMOOTHING = 3.0
    CONTOUR_LINE_WIDTH = 2
    
    # Vector field
    VECTOR_GRID_SPACING = 30
    VECTOR_SCALE = 20
    VECTOR_MIN_THRESHOLD = 0.03  # Lower = show more vectors
    
    # Feature constellation
    MAX_FEATURES = 200
    FEATURE_QUALITY = 0.01
    MIN_FEATURE_DISTANCE = 15
    FEATURE_GRID_SAMPLE = True  # Ensure even distribution
    
    # Color extraction
    NUM_COLORS = 6
    
    # HUD
    GRID_DIVISIONS = 8
    

# =============================================================================
# LAYER 1: LUMINANCE CONTOURS
# =============================================================================

def generate_contours(gray, config=Config):
    """
    Extract topographic-style contour lines from luminance.
    Uses marching squares via OpenCV's findContours on thresholded bands.
    """
    h, w = gray.shape
    smoothed = gaussian_filter(gray.astype(float), sigma=config.CONTOUR_SMOOTHING)
    
    contours_all = []
    # Wider range for better coverage
    levels = np.linspace(15, 240, config.CONTOUR_LEVELS)
    
    for level in levels:
        binary = (smoothed > level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter tiny contours but keep more
        contours = [c for c in contours if cv2.arcLength(c, True) > 30]
        contours_all.append((level, contours))
    
    return contours_all


def draw_contours(canvas, contours_all, offset, scale, config=Config):
    """Draw contour lines with varying opacity by level."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    
    for i, (level, contours) in enumerate(contours_all):
        # Higher alpha for better visibility on dark regions
        alpha = int(80 + (level / 255) * 100)
        alpha = min(alpha, 180)
        color = (config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha)
        
        for contour in contours:
            points = contour.squeeze()
            if len(points.shape) == 1:
                continue
            
            # Scale and offset points
            scaled = [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1])) 
                      for p in points]
            
            if len(scaled) > 2:
                draw.line(scaled + [scaled[0]], fill=color, width=config.CONTOUR_LINE_WIDTH)
    
    return canvas


# =============================================================================
# LAYER 2: GRADIENT VECTOR FIELD
# =============================================================================

def compute_gradient_field(gray, config=Config):
    """
    Compute gradient vectors using Sobel operators.
    Returns grid of (x, y, magnitude, angle) tuples.
    """
    # Sobel derivatives
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
    """Draw gradient arrows across the entire image."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    
    for x, y, mag, ang in vectors:
        # Much lower threshold to show vectors everywhere
        if mag < max_mag * config.VECTOR_MIN_THRESHOLD:
            continue
        
        # Normalize magnitude for display
        norm_mag = (mag / max_mag) * config.VECTOR_SCALE
        norm_mag = max(norm_mag, 4)  # Minimum visible length
        
        # Calculate arrow endpoint
        sx = int(x * scale + offset[0])
        sy = int(y * scale + offset[1])
        ex = int(sx + np.cos(ang) * norm_mag)
        ey = int(sy + np.sin(ang) * norm_mag)
        
        # Alpha based on magnitude but with a floor
        alpha = int(50 + (mag / max_mag) * 150)
        alpha = min(alpha, 200)
        color = (config.SECONDARY[0], config.SECONDARY[1], config.SECONDARY[2], alpha)
        
        draw.line([(sx, sy), (ex, ey)], fill=color, width=1)
        
        # Small dot at origin
        draw.ellipse([sx-1, sy-1, sx+1, sy+1], fill=color)
    
    return canvas


# =============================================================================
# LAYER 3: FREQUENCY SPECTRUM (FFT)
# =============================================================================

def compute_fft_spectrum(gray):
    """
    Compute 2D FFT magnitude spectrum.
    Returns log-scaled, shifted spectrum image with enhanced contrast.
    """
    # Apply window to reduce edge artifacts
    h, w = gray.shape
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window = np.outer(window_y, window_x)
    
    windowed = gray.astype(float) * window
    
    # FFT
    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Log scale for visibility
    log_magnitude = np.log1p(magnitude)
    
    # Normalize
    log_magnitude = log_magnitude / log_magnitude.max()
    
    # Strong gamma correction to bring out details
    gamma = 0.3
    log_magnitude = np.power(log_magnitude, gamma)
    
    # Contrast stretch
    p_low, p_high = np.percentile(log_magnitude, [1, 99])
    log_magnitude = np.clip((log_magnitude - p_low) / (p_high - p_low + 1e-8), 0, 1)
    
    return (log_magnitude * 255).astype(np.uint8)


def draw_fft_inset(canvas, spectrum, position, size, config=Config):
    """Draw FFT spectrum as a small inset panel with enhanced visibility."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size
    
    # Resize spectrum
    spec_img = Image.fromarray(spectrum)
    spec_img = spec_img.resize((w - 4, h - 4), Image.Resampling.LANCZOS)
    spec_array = np.array(spec_img)
    
    # Create RGB image with cyan/teal coloring
    colored = np.zeros((spec_array.shape[0], spec_array.shape[1], 4), dtype=np.uint8)
    
    # Use spectrum values directly for intensity, tint with primary color
    intensity = spec_array.astype(float) / 255.0
    colored[:, :, 0] = (intensity * config.PRIMARY[0]).astype(np.uint8)
    colored[:, :, 1] = (intensity * config.PRIMARY[1]).astype(np.uint8)
    colored[:, :, 2] = (intensity * config.PRIMARY[2]).astype(np.uint8)
    colored[:, :, 3] = (intensity * 255).astype(np.uint8)
    
    # Add a base layer so it's visible on white background
    base = np.zeros((spec_array.shape[0], spec_array.shape[1], 4), dtype=np.uint8)
    base[:, :, 0] = 240
    base[:, :, 1] = 245
    base[:, :, 2] = 250
    base[:, :, 3] = 255
    
    base_img = Image.fromarray(base, 'RGBA')
    spec_colored = Image.fromarray(colored, 'RGBA')
    
    # Draw frame with light fill
    draw.rectangle([x, y, x + w, y + h], fill=(245, 248, 250, 255), outline=config.PRIMARY, width=1)
    
    # Composite spectrum onto base
    combined = Image.alpha_composite(base_img, spec_colored)
    canvas.paste(combined, (x + 2, y + 2))
    
    # Label
    draw.text((x + 4, y + h + 4), "FREQ SPECTRUM [FFT²]", fill=config.DIM)
    
    return canvas


# =============================================================================
# LAYER 4: DOMINANT COLOR SPECTRA
# =============================================================================

def extract_dominant_colors(image, config=Config):
    """
    Extract dominant colors using K-means clustering.
    Returns list of (RGB, percentage) tuples sorted by luminance.
    """
    # Reshape image to pixel list
    pixels = np.array(image).reshape(-1, 3)
    
    # Sample for speed
    sample_size = min(10000, len(pixels))
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    sample = pixels[indices]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=config.NUM_COLORS, random_state=42, n_init=10)
    kmeans.fit(sample)
    
    # Get colors and their frequencies
    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    percentages = counts / counts.sum()
    
    # Sort by luminance
    results = []
    for color, pct in zip(colors, percentages):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        results.append((tuple(color), pct, lum))
    
    results.sort(key=lambda x: x[2])
    return [(r[0], r[1]) for r in results]


def draw_color_spectra(canvas, colors, position, size, config=Config):
    """Draw emission-line style color spectrum on white background."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    x, y = position
    w, h = size
    
    # Frame with light fill
    draw.rectangle([x, y, x + w, y + h], fill=(250, 250, 250, 255), outline=config.PRIMARY, width=1)
    
    # Draw each color as emission line
    bar_height = (h - 20) // len(colors)
    
    for i, (color, pct) in enumerate(colors):
        by = y + 10 + i * bar_height
        bar_width = int((w - 20) * pct * 2)  # Scale for visibility
        bar_width = min(bar_width, w - 20)
        
        # Color bar with subtle shadow
        draw.rectangle(
            [x + 12, by + 2, x + 12 + bar_width, by + bar_height - 2],
            fill=(200, 200, 200, 100)
        )
        draw.rectangle(
            [x + 10, by, x + 10 + bar_width, by + bar_height - 4],
            fill=(color[0], color[1], color[2], 255),
            outline=(color[0]//2, color[1]//2, color[2]//2, 100),
            width=1
        )
        
        # Hex code
        hex_code = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        draw.text((x + w - 70, by), hex_code, fill=config.DIM)
    
    # Label
    draw.text((x + 4, y + h + 4), "CHROMATIC EMISSION", fill=config.DIM)
    
    return canvas


# =============================================================================
# LAYER 5: FEATURE CONSTELLATION
# =============================================================================

def detect_features(gray, config=Config):
    """
    Detect corner features using Harris detector with grid-based distribution.
    Ensures full image coverage by adding fallback points in empty cells.
    """
    h, w = gray.shape
    
    # Harris corner detection
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    
    # Threshold
    threshold = config.FEATURE_QUALITY * harris.max()
    
    # Grid-based sampling for even distribution
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
            
            # Try to find a feature in this cell
            if cell.max() > threshold * 0.5:  # Lower threshold for more coverage
                local_coords = np.unravel_index(cell.argmax(), cell.shape)
                points.append([x_start + local_coords[1], y_start + local_coords[0]])
            else:
                # Fallback: add point at cell center with some jitter
                cx = x_start + cell_w // 2 + np.random.randint(-cell_w//4, cell_w//4)
                cy = y_start + cell_h // 2 + np.random.randint(-cell_h//4, cell_h//4)
                cx = np.clip(cx, x_start, x_end - 1)
                cy = np.clip(cy, y_start, y_end - 1)
                points.append([cx, cy])
    
    return np.array(points) if points else np.array([]).reshape(0, 2)


def compute_delaunay(points):
    """Compute Delaunay triangulation of points."""
    if len(points) < 4:
        return None
    
    try:
        tri = Delaunay(points)
        return tri
    except:
        return None


def draw_constellation(canvas, points, triangulation, offset, scale, config=Config):
    """Draw feature points and Delaunay mesh across the whole image."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    
    if len(points) == 0:
        return canvas
    
    # Scale points
    scaled_points = [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1])) 
                     for p in points]
    
    # Draw triangulation edges
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
            # Distance-based alpha but with higher base
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            alpha = max(20, min(90, int(200 - dist * 0.4)))
            draw.line([p1, p2], fill=(config.TERTIARY[0], config.TERTIARY[1], 
                                       config.TERTIARY[2], alpha), width=1)
    
    # Draw feature points with glow
    for i, (px, py) in enumerate(scaled_points):
        # Outer glow
        draw.ellipse([px-5, py-5, px+5, py+5], 
                     fill=(config.TERTIARY[0], config.TERTIARY[1], config.TERTIARY[2], 30))
        draw.ellipse([px-3, py-3, px+3, py+3], 
                     fill=(config.TERTIARY[0], config.TERTIARY[1], config.TERTIARY[2], 60))
        # Inner point
        draw.ellipse([px-2, py-2, px+2, py+2], fill=config.TERTIARY)
    
    return canvas


# =============================================================================
# LAYER 6: HUD FRAME
# =============================================================================

def draw_hud_frame(canvas, image_bounds, original_size, config=Config):
    """Draw coordinate grid, brackets, scan lines, and technical overlays."""
    draw = ImageDraw.Draw(canvas, 'RGBA')
    
    x, y, w, h = image_bounds
    orig_w, orig_h = original_size
    
    # SCAN GRID - full coverage overlay (darker for white bg)
    grid_lines = 16
    for i in range(1, grid_lines):
        # Vertical scan lines
        lx = x + (w * i // grid_lines)
        alpha = 40 if i % 4 != 0 else 70
        draw.line([(lx, y), (lx, y + h)], fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)
        
        # Horizontal scan lines  
        ly = y + (h * i // grid_lines)
        draw.line([(x, ly), (x + w, ly)], fill=(config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], alpha), width=1)
    
    # Corner brackets
    bracket_len = 30
    bracket_color = config.PRIMARY
    
    # Top-left
    draw.line([(x, y), (x + bracket_len, y)], fill=bracket_color, width=2)
    draw.line([(x, y), (x, y + bracket_len)], fill=bracket_color, width=2)
    
    # Top-right
    draw.line([(x + w, y), (x + w - bracket_len, y)], fill=bracket_color, width=2)
    draw.line([(x + w, y), (x + w, y + bracket_len)], fill=bracket_color, width=2)
    
    # Bottom-left
    draw.line([(x, y + h), (x + bracket_len, y + h)], fill=bracket_color, width=2)
    draw.line([(x, y + h), (x, y + h - bracket_len)], fill=bracket_color, width=2)
    
    # Bottom-right
    draw.line([(x + w, y + h), (x + w - bracket_len, y + h)], fill=bracket_color, width=2)
    draw.line([(x + w, y + h), (x + w, y + h - bracket_len)], fill=bracket_color, width=2)
    
    # Coordinate ticks along edges
    divisions = config.GRID_DIVISIONS
    
    for i in range(divisions + 1):
        # Horizontal ticks (top and bottom)
        tx = x + (w * i // divisions)
        coord_x = orig_w * i // divisions
        
        draw.line([(tx, y - 8), (tx, y - 2)], fill=config.DIM, width=1)
        draw.line([(tx, y + h + 2), (tx, y + h + 8)], fill=config.DIM, width=1)
        
        if i % 2 == 0:
            draw.text((tx - 15, y - 22), f"{coord_x:04d}", fill=config.DIM)
        
        # Vertical ticks (left and right)
        ty = y + (h * i // divisions)
        coord_y = orig_h * i // divisions
        
        draw.line([(x - 8, ty), (x - 2, ty)], fill=config.DIM, width=1)
        draw.line([(x + w + 2, ty), (x + w + 8, ty)], fill=config.DIM, width=1)
        
        if i % 2 == 0:
            draw.text((x - 45, ty - 6), f"{coord_y:04d}", fill=config.DIM)
    
    # Technical readouts in corners
    canvas_w, canvas_h = canvas.size
    
    # Top-left: dimensions
    draw.text((15, 15), f"SOURCE: {orig_w}×{orig_h}px", fill=config.PRIMARY)
    draw.text((15, 32), f"ANALYSIS: ORBITAL SURVEY v1.0", fill=config.DIM)
    
    # Top-right: scan info
    draw.text((canvas_w - 200, 15), "SCAN COMPLETE", fill=config.SECONDARY)
    draw.text((canvas_w - 200, 32), f"GRID: {divisions}×{divisions}", fill=config.DIM)
    
    # Crosshair at center
    cx, cy = x + w // 2, y + h // 2
    cross_size = 20
    cross_color = (config.PRIMARY[0], config.PRIMARY[1], config.PRIMARY[2], 120)
    draw.line([(cx - cross_size, cy), (cx - 5, cy)], fill=cross_color, width=1)
    draw.line([(cx + 5, cy), (cx + cross_size, cy)], fill=cross_color, width=1)
    draw.line([(cx, cy - cross_size), (cx, cy - 5)], fill=cross_color, width=1)
    draw.line([(cx, cy + 5), (cx, cy + cross_size)], fill=cross_color, width=1)
    
    return canvas


# =============================================================================
# MAIN COMPOSITOR
# =============================================================================

def process_image(input_path, output_path=None, config=Config):
    """
    Main processing pipeline.
    Loads image, generates all layers, composites final output.
    """
    print(f"Loading: {input_path}")
    
    # Load image
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Calculate layout
    aspect = orig_w / orig_h
    canvas_w = config.OUTPUT_WIDTH
    
    # Image area (with padding for HUD)
    img_area_w = int(canvas_w * 0.65)
    img_area_h = int(img_area_w / aspect)
    
    # Side panel for insets
    panel_w = canvas_w - img_area_w - config.PADDING * 3
    
    canvas_h = max(img_area_h + config.PADDING * 2, 700)
    
    # Create canvas
    canvas = Image.new('RGBA', (canvas_w, canvas_h), config.BACKGROUND)
    
    # Calculate image placement
    img_x = config.PADDING
    img_y = (canvas_h - img_area_h) // 2
    
    # Resize source image
    img_resized = img.resize((img_area_w, img_area_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Slightly desaturate the base image for overlay visibility
    base_img = Image.fromarray(img_array)
    # Gentle fade to help overlays pop
    base_dark = Image.blend(
        base_img, 
        Image.new('RGB', base_img.size, (255, 255, 255)), 
        alpha=0.15
    )
    canvas.paste(base_dark, (img_x, img_y))
    
    # Scale factor for overlays
    # Since we compute on resized image, coordinates are already in display space
    scale = 1.0  # No additional scaling needed
    offset = (img_x, img_y)
    
    # --- Generate and draw layers ---
    
    print("  [1/6] Computing luminance contours...")
    contours = generate_contours(gray, config)
    canvas = draw_contours(canvas, contours, offset, scale, config)
    
    print("  [2/6] Computing gradient vector field...")
    vectors, max_mag = compute_gradient_field(gray, config)
    canvas = draw_vector_field(canvas, vectors, max_mag, offset, scale, config)
    
    print("  [3/6] Computing frequency spectrum...")
    spectrum = compute_fft_spectrum(gray)
    fft_size = (panel_w, int(panel_w * 0.75))
    fft_pos = (img_x + img_area_w + config.PADDING, img_y)
    canvas = draw_fft_inset(canvas, spectrum, fft_pos, fft_size, config)
    
    print("  [4/6] Extracting dominant colors...")
    colors = extract_dominant_colors(img_resized, config)
    color_pos = (fft_pos[0], fft_pos[1] + fft_size[1] + 40)
    color_size = (panel_w, 180)
    canvas = draw_color_spectra(canvas, colors, color_pos, color_size, config)
    
    print("  [5/6] Detecting features and computing triangulation...")
    features = detect_features(gray, config)
    triangulation = compute_delaunay(features)
    canvas = draw_constellation(canvas, features, triangulation, offset, scale, config)
    
    print("  [6/6] Drawing HUD frame...")
    image_bounds = (img_x, img_y, img_area_w, img_area_h)
    canvas = draw_hud_frame(canvas, image_bounds, (orig_w, orig_h), config)
    
    # Add layer legend at bottom
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
    
    # Stats panel
    stats_x = fft_pos[0]
    stats_y = color_pos[1] + color_size[1] + 40
    draw.text((stats_x, stats_y), "─── ANALYSIS METRICS ───", fill=config.DIM)
    draw.text((stats_x, stats_y + 20), f"Contour levels: {config.CONTOUR_LEVELS}", fill=config.DIM)
    draw.text((stats_x, stats_y + 38), f"Feature points: {len(features)}", fill=config.DIM)
    draw.text((stats_x, stats_y + 56), f"Gradient samples: {len(vectors)}", fill=config.DIM)
    draw.text((stats_x, stats_y + 74), f"Color clusters: {config.NUM_COLORS}", fill=config.DIM)
    
    # Save output
    if output_path is None:
        input_p = Path(input_path)
        output_path = input_p.parent / f"{input_p.stem}_survey{input_p.suffix}"
    
    # Convert to RGB for saving as JPEG/PNG with white background
    final = Image.new('RGB', canvas.size, (255, 255, 255))
    final.paste(canvas, mask=canvas.split()[3])
    
    final.save(output_path, quality=95)
    print(f"\nSaved: {output_path}")
    
    return output_path


# =============================================================================
# CLI WITH TKINTER FILE DIALOG
# =============================================================================

if __name__ == "__main__":
    import sys
    import tkinter as tk
    from tkinter import filedialog
    
    def select_file():
        """Open file dialog to select an image."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring dialog to front
        
        file_path = filedialog.askopenfilename(
            title="Select Image for Orbital Survey",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    
    def select_output():
        """Open file dialog to select output location."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.asksaveasfilename(
            title="Save Survey Result As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    
    # Check command line args first
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Use tkinter dialog
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
    
    result = process_image(input_path, output_path)
    print(f"\nDone! Output saved to: {result}")
