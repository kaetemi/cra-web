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

let pyodide = null;
let wasmReady = false;
let inputImageData = null;
let refImageData = null;
let scriptsLoaded = {};

// Initialize everything
async function init() {
    const overlay = document.getElementById('init-overlay');
    const statusEl = document.getElementById('init-status');
    const progressBar = document.getElementById('init-progress-bar');

    try {
        // Step 1: Load WASM dither module (10%)
        statusEl.textContent = 'Loading WASM dither module...';
        progressBar.style.width = '5%';

        const wasmModule = await import('./wasm/dither.js');
        await wasmModule.default();
        window.floyd_steinberg_dither_wasm = wasmModule.floyd_steinberg_dither;
        wasmReady = true;
        progressBar.style.width = '10%';

        // Step 2: Load Pyodide (60%)
        statusEl.textContent = 'Loading Python runtime (this may take a moment)...';
        progressBar.style.width = '15%';

        pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/'
        });
        progressBar.style.width = '50%';

        // Step 3: Load packages (90%)
        statusEl.textContent = 'Loading image processing packages...';

        await pyodide.loadPackage(['numpy', 'opencv-python', 'pillow', 'scikit-image', 'micropip']);
        progressBar.style.width = '85%';

        // Step 4: Install blendmodes (not needed for these scripts but mentioned in plan)
        statusEl.textContent = 'Finalizing setup...';
        progressBar.style.width = '95%';

        // Done
        progressBar.style.width = '100%';
        statusEl.textContent = 'Ready!';

        await new Promise(resolve => setTimeout(resolve, 500));
        overlay.classList.add('hidden');

        // Enable the process button if we have images
        updateProcessButton();

    } catch (error) {
        statusEl.textContent = 'Error: ' + error.message;
        progressBar.style.background = '#e74c3c';
        console.error('Initialization error:', error);
    }
}

// Load a Python script (stripping out the __main__ block)
async function loadScript(scriptName) {
    if (scriptsLoaded[scriptName]) {
        return;
    }

    const response = await fetch('./scripts/' + scriptName);
    let code = await response.text();

    // Remove the if __name__ == "__main__": block to prevent argparse from running
    // This regex matches the block and everything after it
    code = code.replace(/if\s+__name__\s*==\s*["']__main__["']:\s*[\s\S]*$/, '');

    await pyodide.runPythonAsync(code);
    scriptsLoaded[scriptName] = true;
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

// Update process button state
function updateProcessButton() {
    const btn = document.getElementById('process-btn');
    btn.disabled = !pyodide || !wasmReady || !inputImageData || !refImageData;
}

// Process images
async function processImages() {
    const method = document.getElementById('method-select').value;
    const config = METHOD_CONFIG[method];
    const loading = document.getElementById('loading');
    const processBtn = document.getElementById('process-btn');
    const errorMessage = document.getElementById('error-message');
    const statusMessage = document.getElementById('processing-status');
    const outputSection = document.getElementById('output-section');

    errorMessage.classList.remove('visible');
    loading.classList.add('active');
    processBtn.disabled = true;
    outputSection.style.display = 'none';

    try {
        // Load the script
        statusMessage.textContent = 'Loading color correction script...';
        await loadScript(config.script);

        // Write images to Pyodide filesystem
        statusMessage.textContent = 'Preparing images...';
        pyodide.FS.writeFile('/input.png', inputImageData);
        pyodide.FS.writeFile('/ref.png', refImageData);

        // Build the main call based on script and options
        statusMessage.textContent = 'Processing (this may take a while)...';

        let pythonCode;
        if (config.script === 'color_correction_basic.py') {
            pythonCode = `main('/input.png', '/ref.png', '/output.png', keep_luminosity=False, verbose=True)`;
        } else if (config.script === 'color_correction_basic_rgb.py') {
            pythonCode = `main('/input.png', '/ref.png', '/output.png', verbose=True)`;
        } else if (config.script === 'color_correction_cra.py') {
            pythonCode = `main('/input.png', '/ref.png', '/output.png', keep_luminosity=False, verbose=True)`;
        } else if (config.script === 'color_correction_tiled.py') {
            const tiledLum = config.options.tiled_luminosity ? 'True' : 'False';
            pythonCode = `main('/input.png', '/ref.png', '/output.png', tiled_luminosity=${tiledLum}, verbose=True)`;
        } else if (config.script === 'color_correction_cra_rgb.py') {
            const perceptual = config.options.perceptual ? 'True' : 'False';
            pythonCode = `main('/input.png', '/ref.png', '/output.png', verbose=True, use_perceptual=${perceptual})`;
        }

        await pyodide.runPythonAsync(pythonCode);

        // Read output
        statusMessage.textContent = 'Reading result...';
        const outputData = pyodide.FS.readFile('/output.png');

        // Create blob and display
        const blob = new Blob([outputData], { type: 'image/png' });
        const url = URL.createObjectURL(blob);

        // Show comparison
        document.getElementById('compare-input').src = document.getElementById('input-preview').src;
        document.getElementById('compare-output').src = url;
        document.getElementById('compare-ref').src = document.getElementById('ref-preview').src;
        document.getElementById('download-btn').href = url;

        outputSection.style.display = 'block';

    } catch (error) {
        console.error('Processing error:', error);
        errorMessage.textContent = 'Error: ' + error.message;
        errorMessage.classList.add('visible');
    } finally {
        loading.classList.remove('active');
        updateProcessButton();
    }
}

// Setup event listeners
document.addEventListener('DOMContentLoaded', () => {
    handleFileUpload('input-file', 'input-preview', 'input-upload', true);
    handleFileUpload('ref-file', 'ref-preview', 'ref-upload', false);

    document.getElementById('method-select').addEventListener('change', updateMethodDescription);
    document.getElementById('process-btn').addEventListener('click', processImages);

    // Start initialization
    init();
});
