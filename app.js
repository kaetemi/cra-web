// Implementation descriptions
const IMPL_DESCRIPTIONS = {
    'wasm': 'Pure Rust implementation compiled to WebAssembly. Fast and runs entirely in the browser.',
    'python': 'Python implementation running via Pyodide. Slower but matches the original scripts exactly.'
};

// Method descriptions
const METHOD_DESCRIPTIONS = {
    'lab': 'Basic LAB histogram matching. Converts to LAB color space and matches histograms for each channel independently. Fast but may cause color flips on axis-aligned colors.',
    'rgb': 'Basic RGB histogram matching in linear RGB space. Simple approach that matches each RGB channel independently. Can cause color shifts.',
    'cra_lab': 'Chroma Rotation Averaging in LAB color space. Rotates the AB chroma plane at multiple angles (0°, 30°, 60°), performs histogram matching at each rotation, then averages the results. This prevents color flips and preserves complex color relationships.',
    'cra_lab_tiled': 'CRA with overlapping tile-based processing. Divides the image into blocks with 50% overlap, applies CRA to each block, then blends results using Hamming windows. Best for images with spatially varying color casts. Includes tiled luminosity processing.',
    'cra_lab_tiled_ab': 'CRA tiled processing for AB chroma channels with global luminosity matching. Applies per-block CRA correction to the A and B channels only, then performs a final global histogram match on all LAB channels including luminosity. Best balance of localized color correction with consistent global tone.',
    'cra_rgb': 'Chroma Rotation Averaging in RGB space. Rotates the RGB cube around the neutral gray axis (1,1,1) using Rodrigues\' rotation formula at 0°, 40°, 80°. Works well when you want to stay in RGB space.',
    'cra_rgb_perceptual': 'CRA RGB with perceptual weighting. Scales channels by Rec.709 luminance weights before rotation, giving more importance to green (which humans perceive most sensitively) and less to blue. Can produce more natural-looking results.'
};

// Map method values to script files and options
const METHOD_CONFIG = {
    'lab': { script: 'color_correction_basic.py', options: {} },
    'rgb': { script: 'color_correction_basic_rgb.py', options: {} },
    'cra_lab': { script: 'color_correction_cra.py', options: {} },
    'cra_lab_tiled': { script: 'color_correction_tiled.py', options: { tiled_luminosity: true } },
    'cra_lab_tiled_ab': { script: 'color_correction_tiled.py', options: { tiled_luminosity: false } },
    'cra_rgb': { script: 'color_correction_cra_rgb.py', options: { perceptual: false } },
    'cra_rgb_perceptual': { script: 'color_correction_cra_rgb.py', options: { perceptual: true } }
};

let worker = null;
let workerReady = false;
let inputImageData = null;
let refImageData = null;
let consoleOutput = null;

// Format and append a line to console output
function appendConsole(text) {
    if (!consoleOutput) return;

    // Count leading spaces to determine indent level
    const match = text.match(/^(\s*)/);
    const spaces = match ? match[1].length : 0;
    const indentLevel = Math.min(Math.floor(spaces / 2), 3);

    const line = document.createElement('div');
    line.className = `indent-${indentLevel}`;
    line.textContent = text.trimStart();
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// Clear console output
function clearConsole() {
    if (consoleOutput) {
        consoleOutput.innerHTML = '';
    }
}

// Initialize worker
function initWorker() {
    const overlay = document.getElementById('init-overlay');
    const statusEl = document.getElementById('init-status');
    const progressBar = document.getElementById('init-progress-bar');

    worker = new Worker('./worker.js');

    worker.onmessage = function(e) {
        const { type, ...data } = e.data;

        switch (type) {
            case 'progress':
                statusEl.textContent = data.message;
                progressBar.style.width = data.percent + '%';
                break;

            case 'ready':
                workerReady = true;
                overlay.classList.add('hidden');
                updateProcessButton();
                break;

            case 'console':
                appendConsole(data.text);
                break;

            case 'error':
                if (!workerReady) {
                    statusEl.textContent = 'Error: ' + data.message;
                    progressBar.style.background = '#e74c3c';
                } else {
                    document.getElementById('error-message').textContent = 'Error: ' + data.message;
                    document.getElementById('error-message').classList.add('visible');
                    document.getElementById('loading').classList.remove('active');
                    updateProcessButton();
                }
                console.error('Worker error:', data.message);
                break;

            case 'complete':
                handleProcessingComplete(data.outputData);
                break;
        }
    };

    worker.onerror = function(error) {
        console.error('Worker error:', error);
        statusEl.textContent = 'Error: ' + error.message;
        progressBar.style.background = '#e74c3c';
    };

    // Start initialization
    worker.postMessage({ type: 'init' });
}

// Handle processing completion
function handleProcessingComplete(outputData) {
    const blob = new Blob([outputData], { type: 'image/png' });
    const url = URL.createObjectURL(blob);

    // Show comparison
    document.getElementById('compare-input').src = document.getElementById('input-preview').src;
    document.getElementById('compare-output').src = url;
    document.getElementById('compare-ref').src = document.getElementById('ref-preview').src;
    document.getElementById('download-btn').href = url;

    document.getElementById('output-section').style.display = 'block';
    document.getElementById('loading').classList.remove('active');
    updateProcessButton();
}

// Handle file upload
function handleFileUpload(inputId, previewId, uploadBoxId, isInput) {
    const fileInput = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const uploadBox = document.getElementById(uploadBoxId);

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Show preview
        const reader = new FileReader();
        reader.onload = async (event) => {
            preview.src = event.target.result;
            preview.style.display = 'block';
            uploadBox.classList.add('has-image');

            // Hide hints
            uploadBox.querySelectorAll('.hint').forEach(el => el.style.display = 'none');

            // Store image data
            const arrayBuffer = await file.arrayBuffer();
            const uint8Array = new Uint8Array(arrayBuffer);

            if (isInput) {
                inputImageData = uint8Array;
            } else {
                refImageData = uint8Array;
            }

            updateProcessButton();
        };
        reader.readAsDataURL(file);
    });

    // Drag and drop support
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#3498db';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = uploadBox.classList.contains('has-image') ? '#27ae60' : '#ccc';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        }
        uploadBox.style.borderColor = uploadBox.classList.contains('has-image') ? '#27ae60' : '#ccc';
    });
}

// Update method description
function updateMethodDescription() {
    const select = document.getElementById('method-select');
    const description = document.getElementById('method-description');
    description.textContent = METHOD_DESCRIPTIONS[select.value];
}

// Update implementation toggle label and description
function updateImplementationLabel() {
    const useWasm = document.getElementById('use-wasm').checked;
    const label = document.getElementById('impl-label');
    const description = document.getElementById('impl-description');
    const f32HistogramSection = document.getElementById('f32-histogram-section');

    if (useWasm) {
        label.textContent = 'WASM (Rust)';
        description.textContent = IMPL_DESCRIPTIONS.wasm;
        // Show f32 histogram toggle when WASM is enabled
        if (f32HistogramSection) {
            f32HistogramSection.style.display = 'block';
        }
    } else {
        label.textContent = 'Python (Pyodide)';
        description.textContent = IMPL_DESCRIPTIONS.python;
        // Hide f32 histogram toggle when Python is selected
        if (f32HistogramSection) {
            f32HistogramSection.style.display = 'none';
        }
    }
}

// Update histogram toggle label and description
function updateHistogramLabel() {
    const useF32Histogram = document.getElementById('use-f32-histogram').checked;
    const label = document.getElementById('histogram-label');
    const description = document.getElementById('histogram-description');

    if (useF32Histogram) {
        label.textContent = 'F32 Quantile';
        description.textContent = 'Use f32 sort-based quantile matching. No quantization noise, works directly on floating point values with linear interpolation.';
    } else {
        label.textContent = 'Binned (256)';
        description.textContent = 'Use binned histogram matching with 256 bins (original/reference). Toggle for f32 sort-based quantile matching (no quantization noise).';
    }
}

// Update process button state
function updateProcessButton() {
    const btn = document.getElementById('process-btn');
    btn.disabled = !workerReady || !inputImageData || !refImageData;
}

// Load default images
async function loadDefaultImages() {
    try {
        // Load input image
        const inputResponse = await fetch('./assets/forest_plain.png');
        const inputBlob = await inputResponse.blob();
        const inputArrayBuffer = await inputBlob.arrayBuffer();
        inputImageData = new Uint8Array(inputArrayBuffer);

        const inputPreview = document.getElementById('input-preview');
        inputPreview.src = URL.createObjectURL(inputBlob);
        inputPreview.style.display = 'block';
        const inputBox = document.getElementById('input-upload');
        inputBox.classList.add('has-image');
        inputBox.querySelectorAll('.hint').forEach(el => el.style.display = 'none');

        // Load reference image
        const refResponse = await fetch('./assets/flowers_golden.png');
        const refBlob = await refResponse.blob();
        const refArrayBuffer = await refBlob.arrayBuffer();
        refImageData = new Uint8Array(refArrayBuffer);

        const refPreview = document.getElementById('ref-preview');
        refPreview.src = URL.createObjectURL(refBlob);
        refPreview.style.display = 'block';
        const refBox = document.getElementById('ref-upload');
        refBox.classList.add('has-image');
        refBox.querySelectorAll('.hint').forEach(el => el.style.display = 'none');

        updateProcessButton();
    } catch (e) {
        console.log('Could not load default images:', e);
    }
}

// Process images using worker
function processImages() {
    const method = document.getElementById('method-select').value;
    const config = METHOD_CONFIG[method];
    const useWasm = document.getElementById('use-wasm').checked;
    const useF32Histogram = document.getElementById('use-f32-histogram')?.checked || false;
    const loading = document.getElementById('loading');
    const processBtn = document.getElementById('process-btn');
    const errorMessage = document.getElementById('error-message');
    const statusMessage = document.getElementById('processing-status');
    const outputSection = document.getElementById('output-section');
    consoleOutput = document.getElementById('console-output');

    errorMessage.classList.remove('visible');
    loading.classList.add('active');
    processBtn.disabled = true;
    outputSection.style.display = 'none';
    clearConsole();

    const implName = useWasm ? 'WASM' : 'Python';
    statusMessage.textContent = `Running color correction (${implName})...`;

    // Send processing request to worker
    worker.postMessage({
        type: 'process',
        inputData: inputImageData,
        refData: refImageData,
        method: method,
        config: config,
        useWasm: useWasm,
        useF32Histogram: useF32Histogram
    });
}

// Register service worker for offline support
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js')
        .then((reg) => console.log('Service worker registered'))
        .catch((err) => console.log('Service worker registration failed:', err));
}

// Setup event listeners
document.addEventListener('DOMContentLoaded', () => {
    handleFileUpload('input-file', 'input-preview', 'input-upload', true);
    handleFileUpload('ref-file', 'ref-preview', 'ref-upload', false);

    document.getElementById('method-select').addEventListener('change', updateMethodDescription);
    document.getElementById('use-wasm').addEventListener('change', updateImplementationLabel);
    document.getElementById('use-f32-histogram').addEventListener('change', updateHistogramLabel);
    document.getElementById('process-btn').addEventListener('click', processImages);

    // Load default images
    loadDefaultImages();

    // Start worker initialization
    initWorker();
});
