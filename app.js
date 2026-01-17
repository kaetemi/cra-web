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
    'cra_rgb_perceptual': 'CRA RGB with perceptual weighting. Scales channels by Rec.709 luminance weights before rotation, giving more importance to green (which humans perceive most sensitively) and less to blue. Can produce more natural-looking results.',
    'oklab': 'Basic Oklab histogram matching. Oklab is a perceptually uniform color space (2020) with better hue linearity than LAB. Matches histograms for L, a, b channels independently. Modern alternative to LAB with more predictable color behavior.',
    'cra_oklab': 'Chroma Rotation Averaging in Oklab color space. Combines CRA\'s rotation averaging with Oklab\'s superior perceptual uniformity. Rotates the AB chroma plane at multiple angles, performs histogram matching, then averages results. Best choice for perceptually accurate color transfer.',
    'cra_oklab_tiled': 'CRA Oklab with overlapping tile-based processing. Divides the image into blocks with 50% overlap, applies CRA in Oklab space to each block, then blends results using Hamming windows. Best for images with spatially varying color casts. Includes tiled luminosity processing.',
    'cra_oklab_tiled_ab': 'CRA Oklab tiled processing for AB chroma channels with global luminosity matching. Applies per-block CRA correction in Oklab to the A and B channels only, then performs a final global histogram match. Best balance of localized color correction with consistent global tone.'
};

// Map method values to script files and options
const METHOD_CONFIG = {
    'lab': { script: 'color_correction_basic.py', options: {} },
    'rgb': { script: 'color_correction_basic_rgb.py', options: {} },
    'cra_lab': { script: 'color_correction_cra.py', options: {} },
    'cra_lab_tiled': { script: 'color_correction_tiled.py', options: { tiled_luminosity: true } },
    'cra_lab_tiled_ab': { script: 'color_correction_tiled.py', options: { tiled_luminosity: false } },
    'cra_rgb': { script: 'color_correction_cra_rgb.py', options: { perceptual: false } },
    'cra_rgb_perceptual': { script: 'color_correction_cra_rgb.py', options: { perceptual: true } },
    'oklab': { script: null, options: {}, wasmOnly: true },
    'cra_oklab': { script: null, options: {}, wasmOnly: true },
    'cra_oklab_tiled': { script: null, options: { tiled_luminosity: true }, wasmOnly: true },
    'cra_oklab_tiled_ab': { script: null, options: { tiled_luminosity: false }, wasmOnly: true }
};

let worker = null;
let workerReady = false;
let inputImageData = null;
let refImageData = null;
let consoleOutput = null;
let processTimestamp = null; // Timestamp when processing started
let lastOutputWidth = 0;
let lastOutputHeight = 0;

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
                if (data.stage === 'init') {
                    statusEl.textContent = data.message;
                    progressBar.style.width = data.percent + '%';
                } else if (data.stage === 'process') {
                    // Update processing progress UI
                    const processingStatus = document.getElementById('processing-status');
                    const processingProgressBar = document.getElementById('progress-bar');
                    const processingProgressText = document.getElementById('progress-text');
                    if (processingStatus) processingStatus.textContent = data.message;
                    if (processingProgressBar) processingProgressBar.style.width = data.percent + '%';
                    if (processingProgressText) processingProgressText.textContent = data.percent + '%';
                }
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

// Generate CLI command based on current settings
function generateCliCommand() {
    const method = document.getElementById('method-select').value;
    const histogramMode = parseInt(document.getElementById('histogram-mode-select')?.value || '0', 10);
    const outputDitherMode = parseInt(document.getElementById('output-dither-select')?.value || '0', 10);
    const histogramDitherMode = parseInt(document.getElementById('histogram-dither-select')?.value || '4', 10);
    const colorAwareHistogram = document.getElementById('color-aware-histogram')?.checked || false;
    const histogramDistanceSpace = parseInt(document.getElementById('histogram-distance-space')?.value || '1', 10);
    const colorAwareOutput = document.getElementById('color-aware-output')?.checked || false;
    const outputDistanceSpace = parseInt(document.getElementById('output-distance-space')?.value || '1', 10);

    // Map method values to CLI method names
    const methodMap = {
        'lab': 'basic-lab',
        'rgb': 'basic-rgb',
        'oklab': 'basic-oklab',
        'cra_lab': 'cra-lab',
        'cra_lab_tiled': 'tiled-lab',
        'cra_lab_tiled_ab': 'tiled-lab',
        'cra_rgb': 'cra-rgb',
        'cra_rgb_perceptual': 'cra-rgb',
        'cra_oklab': 'cra-oklab',
        'cra_oklab_tiled': 'tiled-oklab',
        'cra_oklab_tiled_ab': 'tiled-oklab'
    };

    // Map dither mode values to CLI dither names
    const ditherMap = {
        0: 'fs-standard',
        1: 'fs-serpentine',
        2: 'jjn-standard',
        3: 'jjn-serpentine',
        4: 'mixed-standard',
        5: 'mixed-serpentine',
        6: 'mixed-random'
    };

    // Map histogram mode values to CLI histogram mode names
    const histogramModeMap = {
        0: 'binned',
        1: 'f32-endpoint',
        2: 'f32-midpoint'
    };

    // Map perceptual space values to CLI names
    const spaceMap = {
        0: 'lab-cie76',
        1: 'oklab',
        2: 'lab-cie94',
        3: 'lab-ciede2000',
        4: 'linear-rgb',
        5: 'y-cb-cr'
    };

    let cmd = `cra -i input.png -r reference.png -o output.png`;
    // Add histogram method only if not default cra-oklab (which is CLI default when reference provided)
    if (method !== 'cra_oklab') {
        cmd += ` --histogram ${methodMap[method] || 'cra-lab'}`;
    }

    // Add keep-luminosity for basic-lab, basic-oklab, cra-lab, cra-oklab if not tiled
    if (['lab', 'oklab', 'cra_lab', 'cra_oklab'].includes(method)) {
        // These methods could have keep_luminosity but web UI doesn't expose it currently
    }

    // Add tiled-luminosity for tiled methods
    if (method === 'cra_lab_tiled' || method === 'cra_oklab_tiled') {
        cmd += ' --tiled-luminosity';
    }

    // Add perceptual for cra-rgb with perceptual weighting
    if (method === 'cra_rgb_perceptual') {
        cmd += ' --perceptual';
    }

    // Add histogram mode (only if not default binned)
    if (histogramMode !== 0) {
        cmd += ` --histogram-mode ${histogramModeMap[histogramMode]}`;
    }

    // Add output dither (only if not default mixed-standard)
    if (outputDitherMode !== 4) {
        cmd += ` --output-dither ${ditherMap[outputDitherMode]}`;
    }

    // Add histogram dither only if binned mode and not default mixed-standard
    if (histogramMode === 0 && histogramDitherMode !== 4) {
        cmd += ` --histogram-dither ${ditherMap[histogramDitherMode]}`;
    }

    // Colorspace-aware histogram options (only applies to binned mode)
    if (histogramMode === 0 && methodSupportsColorAware(method)) {
        if (!colorAwareHistogram) {
            // Colorspace-aware is default ON, add flag only if disabled
            cmd += ' --no-colorspace-aware-histogram';
        } else if (histogramDistanceSpace !== 1) {
            // Add distance space only if non-default (1 = Oklab)
            cmd += ` --histogram-distance-space ${spaceMap[histogramDistanceSpace]}`;
        }
    }

    // Colorspace-aware output is ON by default in CLI
    if (methodSupportsColorAware(method)) {
        if (!colorAwareOutput) {
            // Disable (differs from CLI default)
            cmd += ' --no-colorspace-aware-output';
        }
        if (colorAwareOutput && outputDistanceSpace !== 1) {
            // Add distance space only if non-default (1 = Oklab)
            cmd += ` --output-distance-space ${spaceMap[outputDistanceSpace]}`;
        }
    }

    return cmd;
}

// Handle processing completion
function handleProcessingComplete(outputData) {
    const blob = new Blob([outputData], { type: 'image/png' });
    const url = URL.createObjectURL(blob);

    // Show comparison
    document.getElementById('compare-input').src = document.getElementById('input-preview').src;
    document.getElementById('compare-output').src = url;
    document.getElementById('compare-ref').src = document.getElementById('ref-preview').src;

    // Get output dimensions from the blob by loading it as an image
    const img = new Image();
    img.onload = function() {
        lastOutputWidth = img.width;
        lastOutputHeight = img.height;
        // Update download filename with dimensions
        const downloadBtn = document.getElementById('download-btn');
        downloadBtn.href = url;
        downloadBtn.download = generateDownloadFilename(lastOutputWidth, lastOutputHeight, 'RGB8', 'png');
    };
    img.src = url;

    // Set initial download link (will be updated with dimensions once image loads)
    const downloadBtn = document.getElementById('download-btn');
    downloadBtn.href = url;

    // Generate and display CLI command
    const cliCommand = generateCliCommand();
    document.getElementById('cli-command').textContent = cliCommand;

    document.getElementById('output-section').style.display = 'block';
    document.getElementById('loading').classList.remove('active');
    updateProcessButton();
}

// Handle file upload
function handleFileUpload(inputId, previewId, uploadBoxId, isInput) {
    const fileInput = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const previewWrapper = document.getElementById(previewId + '-wrapper');
    const uploadBox = document.getElementById(uploadBoxId);

    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Show preview
        const reader = new FileReader();
        reader.onload = async (event) => {
            preview.src = event.target.result;
            previewWrapper.style.display = 'inline-block';
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
    updateColorAwareSectionsVisibility();
    updateProcessButton();
}

// Update implementation toggle label and description
function updateImplementationLabel() {
    const useWasm = document.getElementById('use-wasm').checked;
    const label = document.getElementById('impl-label');
    const description = document.getElementById('impl-description');
    const histogramModeSection = document.getElementById('histogram-mode-section');
    const outputDitherSection = document.getElementById('output-dither-section');
    const histogramDitherSection = document.getElementById('histogram-dither-section');
    const colorAwareHistogramSection = document.getElementById('color-aware-histogram-section');
    const colorAwareOutputSection = document.getElementById('color-aware-output-section');

    if (useWasm) {
        label.textContent = 'WASM (Rust)';
        description.textContent = IMPL_DESCRIPTIONS.wasm;
        // Show histogram mode dropdown when WASM is enabled
        if (histogramModeSection) {
            histogramModeSection.style.display = 'block';
        }
        // Show output dither section when WASM is enabled
        if (outputDitherSection) {
            outputDitherSection.style.display = 'block';
        }
        // Show/hide histogram dither based on histogram mode setting
        updateHistogramModeDescription();
        // Update color-aware sections visibility
        updateColorAwareSectionsVisibility();
    } else {
        label.textContent = 'Python (Pyodide)';
        description.textContent = IMPL_DESCRIPTIONS.python;
        // Hide histogram mode dropdown when Python is selected
        if (histogramModeSection) {
            histogramModeSection.style.display = 'none';
        }
        // Hide dithering options when Python is selected (not supported)
        if (outputDitherSection) {
            outputDitherSection.style.display = 'none';
        }
        if (histogramDitherSection) {
            histogramDitherSection.style.display = 'none';
        }
        // Hide color-aware sections when Python is selected
        if (colorAwareHistogramSection) {
            colorAwareHistogramSection.style.display = 'none';
        }
        if (colorAwareOutputSection) {
            colorAwareOutputSection.style.display = 'none';
        }
    }
    updateProcessButton();
}

// Histogram mode descriptions
const HISTOGRAM_MODE_DESCRIPTIONS = {
    '0': 'Binned histogram matching with 256 bins. Classic approach with dithering for quantization.',
    '1': 'F32 sort-based quantile matching with endpoint alignment. Rank 0 maps to ref[0], rank n-1 maps to ref[m-1]. Preserves reference extremes.',
    '2': 'F32 sort-based quantile matching with midpoint alignment. Statistically correct quantile sampling, avoids color expansion bias at extremes.'
};

// Update histogram mode description and show/hide histogram dither section
function updateHistogramModeDescription() {
    const histogramMode = document.getElementById('histogram-mode-select').value;
    const description = document.getElementById('histogram-mode-description');
    const histogramDitherSection = document.getElementById('histogram-dither-section');
    const colorAwareHistogramSection = document.getElementById('color-aware-histogram-section');

    description.textContent = HISTOGRAM_MODE_DESCRIPTIONS[histogramMode] || HISTOGRAM_MODE_DESCRIPTIONS['0'];

    // Show histogram dither section and color-aware options only when using binned histogram (mode 0)
    const isBinned = histogramMode === '0';
    if (histogramDitherSection) {
        histogramDitherSection.style.display = isBinned ? 'block' : 'none';
    }
    if (colorAwareHistogramSection) {
        colorAwareHistogramSection.style.display = isBinned ? 'block' : 'none';
    }
}

// Check if current method supports color-aware options (Lab/Oklab CRA methods)
function methodSupportsColorAware(method) {
    return ['cra_lab', 'cra_lab_tiled', 'cra_lab_tiled_ab',
            'cra_oklab', 'cra_oklab_tiled', 'cra_oklab_tiled_ab'].includes(method);
}

// Update color-aware histogram toggle
function updateColorAwareHistogramToggle() {
    const checkbox = document.getElementById('color-aware-histogram');
    const label = document.getElementById('color-aware-histogram-label');
    const distanceContainer = document.getElementById('histogram-distance-space-container');

    if (checkbox.checked) {
        label.textContent = 'On';
        if (distanceContainer) distanceContainer.style.display = 'block';
    } else {
        label.textContent = 'Off';
        if (distanceContainer) distanceContainer.style.display = 'none';
    }
}

// Update color-aware output toggle
function updateColorAwareOutputToggle() {
    const checkbox = document.getElementById('color-aware-output');
    const label = document.getElementById('color-aware-output-label');
    const distanceContainer = document.getElementById('output-distance-space-container');

    if (checkbox.checked) {
        label.textContent = 'On';
        if (distanceContainer) distanceContainer.style.display = 'block';
    } else {
        label.textContent = 'Off';
        if (distanceContainer) distanceContainer.style.display = 'none';
    }
}

// Update visibility of color-aware sections based on method and WASM mode
function updateColorAwareSectionsVisibility() {
    const useWasm = document.getElementById('use-wasm').checked;
    const method = document.getElementById('method-select').value;
    const histogramMode = document.getElementById('histogram-mode-select').value;
    const colorAwareHistogramSection = document.getElementById('color-aware-histogram-section');
    const colorAwareOutputSection = document.getElementById('color-aware-output-section');

    const showColorAware = useWasm && methodSupportsColorAware(method);
    const isBinned = histogramMode === '0';

    if (colorAwareHistogramSection) {
        colorAwareHistogramSection.style.display = (showColorAware && isBinned) ? 'block' : 'none';
    }
    if (colorAwareOutputSection) {
        colorAwareOutputSection.style.display = showColorAware ? 'block' : 'none';
    }
}

// Output dithering method descriptions
const OUTPUT_DITHER_DESCRIPTIONS = {
    '0': 'Floyd-Steinberg with standard left-to-right scanning. Fast and widely used error diffusion algorithm. Used for final output quantization.',
    '1': 'Floyd-Steinberg with serpentine scanning (alternating direction each row). Reduces diagonal banding artifacts. Used for final output quantization.',
    '2': 'Jarvis-Judice-Ninke with standard scanning. Larger 3-row kernel produces smoother gradients. Used for final output quantization.',
    '3': 'Jarvis-Judice-Ninke with serpentine scanning. Combines larger kernel with alternating scan direction for best quality. Used for final output quantization.',
    '4': 'Mixed: Randomly selects between Floyd-Steinberg and JJN kernels per-pixel with standard scanning. Used for final output quantization.',
    '5': 'Mixed: Randomly selects between Floyd-Steinberg and JJN kernels per-pixel with serpentine scanning. Used for final output quantization.',
    '6': 'Mixed: Randomly selects both kernel AND scan direction per-row. Most randomized option. Used for final output quantization.'
};

// Histogram dithering method descriptions
const HISTOGRAM_DITHER_DESCRIPTIONS = {
    '0': 'Floyd-Steinberg with standard scanning. Used for histogram processing quantization (when not using f32 histogram).',
    '1': 'Floyd-Steinberg with serpentine scanning. Used for histogram processing quantization (when not using f32 histogram).',
    '2': 'Jarvis-Judice-Ninke with standard scanning. Used for histogram processing quantization (when not using f32 histogram).',
    '3': 'Jarvis-Judice-Ninke with serpentine scanning. Used for histogram processing quantization (when not using f32 histogram).',
    '4': 'Mixed: Randomly selects between Floyd-Steinberg and JJN kernels per-pixel. Used for histogram processing quantization (when not using f32 histogram).',
    '5': 'Mixed: Randomly selects between Floyd-Steinberg and JJN kernels per-pixel with serpentine scanning. Used for histogram processing quantization.',
    '6': 'Mixed: Randomly selects both kernel AND scan direction per-row. Used for histogram processing quantization.'
};

// Update output dither method description
function updateOutputDitherDescription() {
    const ditherMode = document.getElementById('output-dither-select').value;
    const description = document.getElementById('output-dither-description');
    description.textContent = OUTPUT_DITHER_DESCRIPTIONS[ditherMode] || OUTPUT_DITHER_DESCRIPTIONS['2'];
}

// Update histogram dither method description
function updateHistogramDitherDescription() {
    const ditherMode = document.getElementById('histogram-dither-select').value;
    const description = document.getElementById('histogram-dither-description');
    description.textContent = HISTOGRAM_DITHER_DESCRIPTIONS[ditherMode] || HISTOGRAM_DITHER_DESCRIPTIONS['4'];
}

// Check if a WASM-only method is selected with Python mode
function isWasmOnlyWithPython() {
    const method = document.getElementById('method-select').value;
    const useWasm = document.getElementById('use-wasm').checked;
    const config = METHOD_CONFIG[method];
    return config && config.wasmOnly && !useWasm;
}

// Update process button state
function updateProcessButton() {
    const btn = document.getElementById('process-btn');
    const wasmOnlyWarning = document.getElementById('wasm-only-warning');
    const isIncompatible = isWasmOnlyWithPython();

    btn.disabled = !workerReady || !inputImageData || !refImageData || isIncompatible;

    // Show/hide warning message
    if (wasmOnlyWarning) {
        wasmOnlyWarning.style.display = isIncompatible ? 'block' : 'none';
    }
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
        const inputPreviewWrapper = document.getElementById('input-preview-wrapper');
        inputPreview.src = URL.createObjectURL(inputBlob);
        inputPreviewWrapper.style.display = 'inline-block';
        const inputBox = document.getElementById('input-upload');
        inputBox.classList.add('has-image');
        inputBox.querySelectorAll('.hint').forEach(el => el.style.display = 'none');

        // Load reference image
        const refResponse = await fetch('./assets/flowers_golden.png');
        const refBlob = await refResponse.blob();
        const refArrayBuffer = await refBlob.arrayBuffer();
        refImageData = new Uint8Array(refArrayBuffer);

        const refPreview = document.getElementById('ref-preview');
        const refPreviewWrapper = document.getElementById('ref-preview-wrapper');
        refPreview.src = URL.createObjectURL(refBlob);
        refPreviewWrapper.style.display = 'inline-block';
        const refBox = document.getElementById('ref-upload');
        refBox.classList.add('has-image');
        refBox.querySelectorAll('.hint').forEach(el => el.style.display = 'none');

        updateProcessButton();
    } catch (e) {
        console.log('Could not load default images:', e);
    }
}

// Generate download filename with timestamp
function generateDownloadFilename(width, height, format, ext) {
    const ts = processTimestamp || Date.now();
    return `cra_${ts}_${width}x${height}_${format}.${ext}`;
}

// Process images using worker
function processImages() {
    const method = document.getElementById('method-select').value;
    const config = METHOD_CONFIG[method];
    const useWasm = document.getElementById('use-wasm').checked;
    // histogramMode: 0 = binned, 1 = f32 endpoint, 2 = f32 midpoint
    const histogramMode = parseInt(document.getElementById('histogram-mode-select')?.value || '0', 10);
    // Default: output=2 (Jarvis), histogram=4 (Mixed)
    const outputDitherMode = parseInt(document.getElementById('output-dither-select')?.value || '2', 10);
    const histogramDitherMode = parseInt(document.getElementById('histogram-dither-select')?.value || '4', 10);
    // Color-aware options
    const colorAwareHistogram = document.getElementById('color-aware-histogram')?.checked || false;
    const histogramDistanceSpace = parseInt(document.getElementById('histogram-distance-space')?.value || '1', 10);
    const colorAwareOutput = document.getElementById('color-aware-output')?.checked || false;
    const outputDistanceSpace = parseInt(document.getElementById('output-distance-space')?.value || '1', 10);

    // Store timestamp when processing starts
    processTimestamp = Date.now();

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

    // Reset progress bar
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    if (progressBar) progressBar.style.width = '0%';
    if (progressText) progressText.textContent = '0%';

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
        histogramMode: histogramMode,
        histogramDitherMode: histogramDitherMode,
        outputDitherMode: outputDitherMode,
        colorAwareHistogram: colorAwareHistogram,
        histogramDistanceSpace: histogramDistanceSpace,
        colorAwareOutput: colorAwareOutput,
        outputDistanceSpace: outputDistanceSpace
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
    document.getElementById('histogram-mode-select').addEventListener('change', () => {
        updateHistogramModeDescription();
        updateColorAwareSectionsVisibility();
    });
    document.getElementById('output-dither-select').addEventListener('change', updateOutputDitherDescription);
    document.getElementById('histogram-dither-select').addEventListener('change', updateHistogramDitherDescription);
    document.getElementById('color-aware-histogram').addEventListener('change', updateColorAwareHistogramToggle);
    document.getElementById('color-aware-output').addEventListener('change', updateColorAwareOutputToggle);
    document.getElementById('process-btn').addEventListener('click', processImages);

    // Initialize histogram dither visibility based on initial histogram mode state
    updateHistogramModeDescription();
    // Initialize color-aware sections visibility
    updateColorAwareSectionsVisibility();

    // Load default images
    loadDefaultImages();

    // Start worker initialization
    initWorker();
});
