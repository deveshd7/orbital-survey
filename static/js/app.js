/**
 * Main application logic
 */

let currentImageFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const exportBtn = document.getElementById('export-btn');
    const newBtn = document.getElementById('new-btn');

    // Click to upload
    uploadBox.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Export button
    exportBtn.addEventListener('click', exportFullImage);

    // New analysis button
    newBtn.addEventListener('click', resetApp);
});

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG, WebP, BMP)');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    currentImageFile = file;
    analyzeImage(file);
}

function analyzeImage(file) {
    // Show loading
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    // Create form data
    const formData = new FormData();
    formData.append('image', file);

    // Send to API
    fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
            resetApp();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing the image');
        resetApp();
    });
}

function displayResults(data) {
    // Hide loading, show results
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';

    // Display image
    const imgContainer = document.getElementById('image-container');
    imgContainer.innerHTML = `<img src="data:image/png;base64,${data.image.base64}" alt="Analyzed Image">`;

    // Image stats
    const statsRow = document.getElementById('image-stats');
    statsRow.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${data.image.original_width}×${data.image.original_height}</div>
            <div class="stat-label">Original Size</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${data.image.width}×${data.image.height}</div>
            <div class="stat-label">Display Size</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${(data.image.original_width / data.image.original_height).toFixed(2)}</div>
            <div class="stat-label">Aspect Ratio</div>
        </div>
    `;

    // Color subtitle
    document.getElementById('color-subtitle').textContent =
        `${data.colors.length} dominant clusters · K-means extraction`;

    // Color chart
    createColorChart(data.colors, '#color-chart');

    // Scatter subtitle
    document.getElementById('scatter-subtitle').textContent =
        `${data.scatter.length} sampled pixels · color space distribution`;

    // Scatter plot
    createScatterPlot(data.scatter, data.colors, '#scatter-plot', '#scatter-tooltip');

    // Stats grid
    const statsGrid = document.getElementById('stats-grid');
    statsGrid.innerHTML = `
        <div class="stat-box">
            <div class="stat-box-value">${data.stats.contour_count}</div>
            <div class="stat-box-label">Contours</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-value">${data.stats.vector_count}</div>
            <div class="stat-box-label">Gradient Vectors</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-value">${data.stats.feature_count}</div>
            <div class="stat-box-label">Features Detected</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-value">${data.stats.avg_gradient.toFixed(1)}</div>
            <div class="stat-box-label">Avg Gradient</div>
        </div>
    `;

    // Scroll to results
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function exportFullImage() {
    if (!currentImageFile) return;

    const exportBtn = document.getElementById('export-btn');
    exportBtn.disabled = true;
    exportBtn.textContent = '⏳ Generating...';

    const formData = new FormData();
    formData.append('image', currentImageFile);

    fetch('/api/export', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        // Download file
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = currentImageFile.name.replace(/\.[^/.]+$/, '') + '_survey_v4.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        exportBtn.disabled = false;
        exportBtn.textContent = '📥 Export Full Analysis (PNG)';
    })
    .catch(error => {
        console.error('Export error:', error);
        alert('Error exporting image');
        exportBtn.disabled = false;
        exportBtn.textContent = '📥 Export Full Analysis (PNG)';
    });
}

function resetApp() {
    currentImageFile = null;
    document.getElementById('upload-section').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    document.getElementById('file-input').value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
