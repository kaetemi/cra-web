// Web Worker for CRA image processing
// Runs Pyodide in a separate thread to keep UI responsive

importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js');

let pyodide = null;
let wasmDitherReady = false;
let wasmCraReady = false;
let craWasm = null;
let scriptsLoaded = {};

// Send progress update to main thread
function sendProgress(stage, message, percent) {
    self.postMessage({ type: 'progress', stage, message, percent });
}

// Send console output to main thread
function sendConsole(text) {
    self.postMessage({ type: 'console', text });
}

// Send error to main thread
function sendError(error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
}

// Send completion to main thread
function sendComplete(outputData) {
    self.postMessage({ type: 'complete', outputData });
}

// Initialize Pyodide and WASM
async function initialize() {
    try {
        // Load WASM dither module (for Python fallback)
        sendProgress('init', 'Loading WASM dither module...', 5);
        const wasmModule = await import('./wasm/dither.js');
        await wasmModule.default();
        self.floyd_steinberg_dither_wasm = wasmModule.floyd_steinberg_dither;
        wasmDitherReady = true;
        sendProgress('init', 'WASM dither module loaded', 8);

        // Load CRA WASM module (full Rust implementation)
        sendProgress('init', 'Loading CRA WASM module...', 10);
        craWasm = await import('./wasm_cra/cra_wasm.js');
        await craWasm.default();
        wasmCraReady = true;
        sendProgress('init', 'CRA WASM module loaded', 15);

        // Load Pyodide
        sendProgress('init', 'Loading Python runtime...', 18);
        pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/'
        });
        sendProgress('init', 'Python runtime loaded', 50);

        // Setup stdout redirect
        pyodide.setStdout({
            batched: (text) => sendConsole(text)
        });

        // Load packages
        sendProgress('init', 'Loading image processing packages...', 55);
        await pyodide.loadPackage(['numpy', 'opencv-python', 'pillow', 'scikit-image', 'micropip']);
        sendProgress('init', 'Packages loaded', 90);

        sendProgress('init', 'Ready!', 100);
        self.postMessage({ type: 'ready' });

    } catch (error) {
        sendError(error);
    }
}

// Track currently active script
let currentScript = null;

// Load a Python script
async function loadScript(scriptName) {
    // Always reload if switching to a different script (each script defines its own main())
    if (currentScript === scriptName && scriptsLoaded[scriptName]) {
        return;
    }

    const response = await fetch('./scripts/' + scriptName);
    let code = await response.text();

    // Remove the if __name__ == "__main__": block
    code = code.replace(/if\s+__name__\s*==\s*["']__main__["']:\s*[\s\S]*$/, '');

    await pyodide.runPythonAsync(code);
    scriptsLoaded[scriptName] = true;
    currentScript = scriptName;
}

// Decode image using WASM (preserves 16-bit precision and ICC profiles)
// Returns: { width, height, hasIcc, is16bit, pixels (f32 0-1), iccProfile (Uint8Array or null) }
function decodeImagePrecise(data) {
    // Decode image to f32 (0-1 normalized, NOT linearized yet)
    const result = craWasm.decode_image_wasm(new Uint8Array(data));
    const width = result[0];
    const height = result[1];
    const hasIcc = result[2] > 0.5;
    const is16bit = result[3] > 0.5;
    const pixels = result.slice(4); // Interleaved RGB f32 (0-1)

    // Extract ICC profile if present
    const iccProfile = hasIcc ? craWasm.extract_icc_profile_wasm(new Uint8Array(data)) : null;

    return { width, height, hasIcc, is16bit, pixels, iccProfile };
}

// Legacy decode using Canvas API (fallback, loses precision)
async function decodeImageCanvas(data) {
    const blob = new Blob([data], { type: 'image/png' });
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return {
        width: bitmap.width,
        height: bitmap.height,
        data: imageData.data // RGBA
    };
}

// Convert RGBA to RGB (remove alpha channel)
function rgbaToRgb(rgbaData) {
    const pixels = rgbaData.length / 4;
    const rgb = new Uint8Array(pixels * 3);
    for (let i = 0; i < pixels; i++) {
        rgb[i * 3] = rgbaData[i * 4];
        rgb[i * 3 + 1] = rgbaData[i * 4 + 1];
        rgb[i * 3 + 2] = rgbaData[i * 4 + 2];
    }
    return rgb;
}

// Encode RGB pixels to PNG
async function encodePng(rgbData, width, height) {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    // Convert RGB to RGBA
    const pixels = width * height;
    for (let i = 0; i < pixels; i++) {
        imageData.data[i * 4] = rgbData[i * 3];
        imageData.data[i * 4 + 1] = rgbData[i * 3 + 1];
        imageData.data[i * 4 + 2] = rgbData[i * 3 + 2];
        imageData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return new Uint8Array(await blob.arrayBuffer());
}

// Process images using WASM with precise 16-bit and ICC support
// Pipeline: file bytes → decode (16-bit preserved) → ICC transform if needed → linear RGB → color_correct → sRGB → dither
async function processImagesWasm(inputData, refData, method, config, histogramMode, histogramDitherMode, outputDitherMode, colorAwareHistogram, histogramDistanceSpace, colorAwareOutput, outputDistanceSpace) {
    sendProgress('process', 'Decoding images (precise)...', 5);

    // Decode with full precision (16-bit + ICC extraction)
    const inputImg = decodeImagePrecise(inputData);
    const refImg = decodeImagePrecise(refData);

    // Log processing info
    sendConsole(`Processing ${inputImg.width}x${inputImg.height} image with WASM...`);
    if (inputImg.is16bit) sendConsole('  Input: 16-bit precision preserved');
    if (inputImg.hasIcc) sendConsole('  Input: ICC profile detected');
    if (refImg.is16bit) sendConsole('  Reference: 16-bit precision preserved');
    if (refImg.hasIcc) sendConsole('  Reference: ICC profile detected');
    sendConsole(`Method: ${method}`);

    const histogramModeNames = {
        0: 'Binned (256 bins)',
        1: 'F32 Endpoint-aligned',
        2: 'F32 Midpoint-aligned'
    };
    sendConsole(`Histogram mode: ${histogramModeNames[histogramMode] || histogramModeNames[0]}`);
    if (histogramMode === 0) {
        const ditherNames = {
            0: 'Floyd-Steinberg (standard)',
            1: 'Floyd-Steinberg (serpentine)',
            2: 'Jarvis-Judice-Ninke (standard)',
            3: 'Jarvis-Judice-Ninke (serpentine)',
            4: 'Mixed (standard)',
            5: 'Mixed (serpentine)',
            6: 'Mixed (random direction)'
        };
        sendConsole(`Histogram dithering: ${ditherNames[histogramDitherMode] || ditherNames[4]}`);
        if (colorAwareHistogram) {
            const distanceSpaceNames = {
                0: 'CIELAB (CIE76)',
                1: 'OkLab',
                2: 'CIELAB (CIE94)',
                3: 'CIELAB (CIEDE2000)',
                4: 'Linear RGB',
                5: "Y'CbCr"
            };
            sendConsole(`Color-aware histogram: ON (${distanceSpaceNames[histogramDistanceSpace] || distanceSpaceNames[1]})`);
        }
    }
    const outputDitherNames = {
        0: 'Floyd-Steinberg (standard)',
        1: 'Floyd-Steinberg (serpentine)',
        2: 'Jarvis-Judice-Ninke (standard)',
        3: 'Jarvis-Judice-Ninke (serpentine)',
        4: 'Mixed (standard)',
        5: 'Mixed (serpentine)',
        6: 'Mixed (random direction)'
    };
    sendConsole(`Output dithering: ${outputDitherNames[outputDitherMode] || outputDitherNames[2]}`);
    if (colorAwareOutput) {
        const distanceSpaceNames = {
            0: 'CIELAB (CIE76)',
            1: 'OkLab',
            2: 'CIELAB (CIE94)',
            3: 'CIELAB (CIEDE2000)',
            4: 'Linear RGB',
            5: "Y'CbCr"
        };
        sendConsole(`Color-aware output: ON (${distanceSpaceNames[outputDistanceSpace] || distanceSpaceNames[1]})`);
    }

    // Method mapping: method name -> [methodCode, luminosityFlag, description]
    const methodMap = {
        'lab':              [0, false, 'basic LAB histogram matching'],
        'rgb':              [1, false, 'basic RGB histogram matching'],
        'oklab':            [2, false, 'basic Oklab histogram matching'],
        'cra_lab':          [3, false, 'CRA LAB color correction'],
        'cra_rgb':          [4, false, 'CRA RGB color correction'],
        'cra_rgb_perceptual': [4, true, 'CRA RGB color correction (perceptual)'],
        'cra_oklab':        [5, false, 'CRA Oklab color correction'],
        'cra_lab_tiled':    [6, true,  'CRA LAB tiled color correction (with tiled luminosity)'],
        'cra_lab_tiled_ab': [6, false, 'CRA LAB tiled color correction (AB only)'],
        'cra_oklab_tiled':  [7, true,  'CRA Oklab tiled color correction (with tiled luminosity)'],
        'cra_oklab_tiled_ab': [7, false, 'CRA Oklab tiled color correction (AB only)'],
    };

    const methodInfo = methodMap[method];
    if (!methodInfo) {
        throw new Error(`Unknown method: ${method}`);
    }

    const [methodCode, luminosityFlag, description] = methodInfo;
    sendConsole(`Running ${description}...`);

    const startTime = performance.now();

    // ========================================================================
    // Precise pipeline: decode → ICC transform (if non-sRGB) → linear → process → sRGB → dither
    // ========================================================================

    sendProgress('process', 'Processing color profiles...', 15);

    // Check if ICC transform is needed for input
    let inputNormalized = inputImg.pixels; // Already f32 0-1 from decode
    if (inputImg.iccProfile && inputImg.iccProfile.length > 0) {
        const isInputSrgb = craWasm.is_icc_profile_srgb_wasm(inputImg.iccProfile);
        if (!isInputSrgb) {
            sendConsole('  Applying ICC transform to input...');
            // Transform directly to linear sRGB
            inputNormalized = craWasm.transform_icc_to_linear_srgb_wasm(
                inputImg.pixels, inputImg.width, inputImg.height, inputImg.iccProfile
            );
            // Skip sRGB->linear below since ICC transform already outputs linear
            var inputAlreadyLinear = true;
        }
    }

    // Check if ICC transform is needed for reference
    let refNormalized = refImg.pixels;
    if (refImg.iccProfile && refImg.iccProfile.length > 0) {
        const isRefSrgb = craWasm.is_icc_profile_srgb_wasm(refImg.iccProfile);
        if (!isRefSrgb) {
            sendConsole('  Applying ICC transform to reference...');
            refNormalized = craWasm.transform_icc_to_linear_srgb_wasm(
                refImg.pixels, refImg.width, refImg.height, refImg.iccProfile
            );
            var refAlreadyLinear = true;
        }
    }

    // Convert to linear if not already done by ICC transform
    sendProgress('process', 'Converting to linear RGB...', 25);
    const inputLinear = inputAlreadyLinear
        ? inputNormalized
        : craWasm.srgb_to_linear_f32_wasm(inputNormalized, inputImg.width, inputImg.height);
    const refLinear = refAlreadyLinear
        ? refNormalized
        : craWasm.srgb_to_linear_f32_wasm(refNormalized, refImg.width, refImg.height);

    // Color correction (in linear RGB space)
    sendProgress('process', 'Running color correction...', 30);
    const resultLinear = craWasm.color_correct_wasm(
        inputLinear, inputImg.width, inputImg.height,
        refLinear, refImg.width, refImg.height,
        methodCode,
        luminosityFlag,
        histogramMode,
        histogramDitherMode,
        colorAwareHistogram,
        histogramDistanceSpace
    );

    // Convert linear back to sRGB (0-1)
    sendProgress('process', 'Converting to sRGB...', 60);
    const resultSrgb_01 = craWasm.linear_to_srgb_f32_wasm(resultLinear, inputImg.width, inputImg.height);

    // Denormalize to 0-255
    sendProgress('process', 'Denormalizing...', 65);
    const resultSrgb_255 = craWasm.denormalize_f32_wasm(resultSrgb_01, inputImg.width, inputImg.height);

    // Dither to RGB888
    sendProgress('process', 'Dithering output...', 70);
    const ditherTechnique = colorAwareOutput ? 2 : 1;
    const resultRgb = craWasm.dither_output_wasm(
        resultSrgb_255,
        inputImg.width, inputImg.height,
        8, 8, 8,  // RGB888
        ditherTechnique,
        outputDitherMode,
        outputDistanceSpace,
        0  // seed
    );

    const elapsed = performance.now() - startTime;
    sendConsole(`Processing completed in ${elapsed.toFixed(0)}ms`);

    sendProgress('process', 'Encoding result...', 80);
    const outputData = await encodePng(resultRgb, inputImg.width, inputImg.height);

    return outputData;
}

// Process images using Python
async function processImagesPython(inputData, refData, method, config) {
    sendProgress('process', 'Loading script...', 0);
    await loadScript(config.script);

    sendProgress('process', 'Preparing images...', 10);
    pyodide.FS.writeFile('/input.png', inputData);
    pyodide.FS.writeFile('/ref.png', refData);

    sendProgress('process', 'Processing...', 20);

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

    sendProgress('process', 'Reading result...', 90);
    const outputData = pyodide.FS.readFile('/output.png');

    return outputData;
}

// Process images (dispatcher)
async function processImages(inputData, refData, method, config, useWasm, histogramMode, histogramDitherMode, outputDitherMode, colorAwareHistogram, histogramDistanceSpace, colorAwareOutput, outputDistanceSpace) {
    try {
        let outputData;

        if (useWasm && wasmCraReady) {
            outputData = await processImagesWasm(inputData, refData, method, config, histogramMode, histogramDitherMode, outputDitherMode, colorAwareHistogram, histogramDistanceSpace, colorAwareOutput, outputDistanceSpace);
        } else {
            outputData = await processImagesPython(inputData, refData, method, config);
        }

        sendComplete(outputData);

    } catch (error) {
        sendError(error);
    }
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { type, ...data } = e.data;

    switch (type) {
        case 'init':
            await initialize();
            break;
        case 'process':
            // Defaults match CLI: histogramMode=0 (binned), histogramDitherMode=4 (Mixed), outputDitherMode=4 (Mixed)
            await processImages(
                data.inputData,
                data.refData,
                data.method,
                data.config,
                data.useWasm,
                data.histogramMode !== undefined ? data.histogramMode : 0,
                data.histogramDitherMode !== undefined ? data.histogramDitherMode : 4,
                data.outputDitherMode !== undefined ? data.outputDitherMode : 4,
                data.colorAwareHistogram !== undefined ? data.colorAwareHistogram : true,
                data.histogramDistanceSpace !== undefined ? data.histogramDistanceSpace : 1,
                data.colorAwareOutput !== undefined ? data.colorAwareOutput : true,
                data.outputDistanceSpace !== undefined ? data.outputDistanceSpace : 1
            );
            break;
    }
};
