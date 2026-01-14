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

// Decode PNG image data to raw RGBA pixels
async function decodeImage(data) {
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

// Process images using WASM
async function processImagesWasm(inputData, refData, method, config, histogramMode, histogramDitherMode, outputDitherMode, colorAwareHistogram, histogramDistanceSpace, colorAwareOutput, outputDistanceSpace) {
    sendProgress('process', 'Decoding images...', 10);
    const inputImg = await decodeImage(inputData);
    const refImg = await decodeImage(refData);

    sendProgress('process', 'Converting to RGB...', 20);
    const inputRgb = rgbaToRgb(inputImg.data);
    const refRgb = rgbaToRgb(refImg.data);

    sendProgress('process', 'Processing with WASM...', 30);
    sendConsole(`Processing ${inputImg.width}x${inputImg.height} image with WASM...`);
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
    // Method codes: 0=BasicLab, 1=BasicRgb, 2=BasicOklab, 3=CraLab, 4=CraRgb, 5=CraOklab, 6=TiledLab, 7=TiledOklab
    // luminosityFlag: keep_luminosity for Lab/Oklab, use_perceptual for CraRgb, tiled_luminosity for Tiled
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
    const resultRgb = craWasm.color_correct_wasm(
        inputRgb, inputImg.width, inputImg.height,
        refRgb, refImg.width, refImg.height,
        methodCode,
        luminosityFlag,
        histogramMode,
        histogramDitherMode,
        colorAwareHistogram,
        histogramDistanceSpace,
        outputDitherMode,
        colorAwareOutput,
        outputDistanceSpace
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
            // Default: histogramMode=0 (binned), histogramDitherMode=4 (Mixed), outputDitherMode=2 (Jarvis)
            await processImages(
                data.inputData,
                data.refData,
                data.method,
                data.config,
                data.useWasm,
                data.histogramMode !== undefined ? data.histogramMode : 0,
                data.histogramDitherMode !== undefined ? data.histogramDitherMode : 4,
                data.outputDitherMode !== undefined ? data.outputDitherMode : 2,
                data.colorAwareHistogram !== undefined ? data.colorAwareHistogram : false,
                data.histogramDistanceSpace !== undefined ? data.histogramDistanceSpace : 1,
                data.colorAwareOutput !== undefined ? data.colorAwareOutput : false,
                data.outputDistanceSpace !== undefined ? data.outputDistanceSpace : 1
            );
            break;
    }
};
