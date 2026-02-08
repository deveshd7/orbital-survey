"""
Flask web app for Orbital Survey V4
Provides JSON API for D3.js frontend
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import orbital_survey_v4 as osv4
import numpy as np
import cv2
from PIL import Image
import colorsys
import io
import base64
from pathlib import Path
import tempfile
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze uploaded image and return JSON data for D3.js visualization
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and process image
        img = Image.open(filepath).convert('RGB')
        orig_w, orig_h = img.size

        # Resize for analysis (same as V4)
        config = osv4.Config()
        img_w = int((config.OUTPUT_WIDTH - config.PADDING * 3) * config.PANEL_SPLIT)
        img_h = int(img_w / (orig_w / orig_h))
        img_resized = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Extract analysis data

        # 1. Color distribution
        colors = osv4.extract_colors(img_resized, config)
        color_data = []
        for rgb, pct, h, s, v in colors:
            color_data.append({
                'rgb': rgb,
                'hex': f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                'percentage': float(pct * 100),
                'hue': float(h * 360),
                'saturation': float(s * 100),
                'value': float(v * 100)
            })

        # 2. Hue-Saturation scatter data
        hsv_data = osv4.compute_hs_scatter(img_resized, sample_count=5000)
        scatter_data = []
        for h, s, v, r, g, b in hsv_data:
            scatter_data.append({
                'hue': float(h * 360),
                'saturation': float(s * 100),
                'value': float(v * 100),
                'rgb': [int(r), int(g), int(b)]
            })

        # 3. Contours (simplified for web - just counts)
        contours = osv4.generate_contours(gray, config)
        contour_count = sum(len(c) for _, c in contours)

        # 4. Gradient stats
        vectors, max_mag = osv4.compute_gradient_field(gray, config)
        avg_magnitude = np.mean([v[2] for v in vectors]) if vectors else 0

        # 5. Features
        features = osv4.detect_features(gray, config)
        feature_data = []
        for point in features:
            feature_data.append({
                'x': float(point[0] / img_w * 100),  # Percentage position
                'y': float(point[1] / img_h * 100)
            })

        # 6. Convert image to base64 for display
        buffered = io.BytesIO()
        img_resized.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Clean up
        os.remove(filepath)

        # Return all data
        return jsonify({
            'success': True,
            'image': {
                'width': img_w,
                'height': img_h,
                'original_width': orig_w,
                'original_height': orig_h,
                'base64': img_base64
            },
            'colors': color_data,
            'scatter': scatter_data,
            'stats': {
                'contour_count': contour_count,
                'vector_count': len(vectors),
                'avg_gradient': float(avg_magnitude),
                'feature_count': len(features)
            },
            'features': feature_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_image():
    """
    Generate full V4 output image and return it
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process with V4
        output_path = filepath.replace('.', '_output.')
        osv4.process_image(filepath, output_path)

        # Send file
        result = send_file(output_path, mimetype='image/png', as_attachment=True,
                          download_name=f"{Path(filename).stem}_survey_v4.png")

        # Clean up
        os.remove(filepath)
        os.remove(output_path)

        return result

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Orbital Survey Web App")
    print("📡 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
