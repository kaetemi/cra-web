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
async function processImagesWasm(inputData, refData, method, config, useF32Histogram, ditherMode) {
    sendProgress('process', 'Decoding images...', 10);
    const inputImg = await decodeImage(inputData);
    const refImg = await decodeImage(refData);

    sendProgress('process', 'Converting to RGB...', 20);
    const inputRgb = rgbaToRgb(inputImg.data);
    const refRgb = rgbaToRgb(refImg.data);

    sendProgress('process', 'Processing with WASM...', 30);
    sendConsole(`Processing ${inputImg.width}x${inputImg.height} image with WASM...`);
    sendConsole(`Method: ${method}`);
    if (useF32Histogram) {
        sendConsole('Using f32 histogram matching (no quantization)');
    }
    if (ditherMode === 1) {
        sendConsole('Using serpentine dithering');
    }

    let resultRgb;
    const startTime = performance.now();

    switch (method) {
        case 'lab':
            sendConsole('Running basic LAB histogram matching...');
            resultRgb = craWasm.color_correct_basic_lab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // keep_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'rgb':
            sendConsole('Running basic RGB histogram matching...');
            resultRgb = craWasm.color_correct_basic_rgb(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_lab':
            sendConsole('Running CRA LAB color correction...');
            resultRgb = craWasm.color_correct_cra_lab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // keep_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_lab_tiled':
            sendConsole('Running CRA LAB tiled color correction (with tiled luminosity)...');
            resultRgb = craWasm.color_correct_tiled_lab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                true, // tiled_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_lab_tiled_ab':
            sendConsole('Running CRA LAB tiled color correction (AB only)...');
            resultRgb = craWasm.color_correct_tiled_lab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // tiled_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_rgb':
            sendConsole('Running CRA RGB color correction...');
            resultRgb = craWasm.color_correct_cra_rgb(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // use_perceptual
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_rgb_perceptual':
            sendConsole('Running CRA RGB color correction (perceptual)...');
            resultRgb = craWasm.color_correct_cra_rgb(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                true, // use_perceptual
                useF32Histogram,
                ditherMode
            );
            break;

        case 'oklab':
            sendConsole('Running basic Oklab histogram matching...');
            resultRgb = craWasm.color_correct_basic_oklab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // keep_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_oklab':
            sendConsole('Running CRA Oklab color correction...');
            resultRgb = craWasm.color_correct_cra_oklab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // keep_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_oklab_tiled':
            sendConsole('Running CRA Oklab tiled color correction (with tiled luminosity)...');
            resultRgb = craWasm.color_correct_tiled_oklab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                true, // tiled_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        case 'cra_oklab_tiled_ab':
            sendConsole('Running CRA Oklab tiled color correction (AB only)...');
            resultRgb = craWasm.color_correct_tiled_oklab(
                inputRgb, inputImg.width, inputImg.height,
                refRgb, refImg.width, refImg.height,
                false, // tiled_luminosity
                useF32Histogram,
                ditherMode
            );
            break;

        default:
            throw new Error(`Unknown method: ${method}`);
    }

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
async function processImages(inputData, refData, method, config, useWasm, useF32Histogram, ditherMode) {
    try {
        let outputData;

        if (useWasm && wasmCraReady) {
            outputData = await processImagesWasm(inputData, refData, method, config, useF32Histogram, ditherMode);
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
            await processImages(data.inputData, data.refData, data.method, data.config, data.useWasm, data.useF32Histogram, data.ditherMode || 0);
            break;
    }
};
