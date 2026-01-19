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

// Decode image using WASM - returns LoadedImage with metadata and pixel access
function decodeImagePrecise(data, preserveAlpha = false) {
    // Single load - keeps u8/u16 pixels, converts on demand (CLI pattern)
    const loadedImage = craWasm.load_image_wasm(new Uint8Array(data));

    return {
        width: loadedImage.width,
        height: loadedImage.height,
        hasIcc: loadedImage.has_non_srgb_icc,  // Only true if non-sRGB
        is16bit: loadedImage.is_16bit,
        hasAlpha: loadedImage.has_alpha,
        isFormatPremultipliedDefault: loadedImage.is_format_premultiplied_default,  // EXR has premultiplied alpha
        // CICP flags (authoritative color space indicators)
        isCicpSrgb: loadedImage.is_cicp_srgb,
        isCicpLinear: loadedImage.is_cicp_linear,
        isCicpNeedsConversion: loadedImage.is_cicp_needs_conversion,
        // Convert to normalized (0-1) for color correction
        // Use RGBA buffer if preserving alpha, otherwise RGB
        buffer: preserveAlpha ? loadedImage.to_normalized_buffer_rgba() : loadedImage.to_normalized_buffer(),
        iccProfile: loadedImage.has_non_srgb_icc ? loadedImage.get_icc_profile() : null,
        // Store loadedImage reference for CICP transform
        loadedImage: loadedImage
    };
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

// Encode RGB or RGBA pixels to PNG
async function encodePng(data, width, height, isRgba = false) {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    const pixels = width * height;
    if (isRgba) {
        // RGBA data - copy directly
        for (let i = 0; i < pixels * 4; i++) {
            imageData.data[i] = data[i];
        }
    } else {
        // RGB data - convert to RGBA with alpha=255
        for (let i = 0; i < pixels; i++) {
            imageData.data[i * 4] = data[i * 3];
            imageData.data[i * 4 + 1] = data[i * 3 + 1];
            imageData.data[i * 4 + 2] = data[i * 3 + 2];
            imageData.data[i * 4 + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    return new Uint8Array(await blob.arrayBuffer());
}

// Process images using WASM with precise 16-bit and ICC support
// Pipeline: file bytes → decode (16-bit preserved) → ICC transform if needed → linear RGB → color_correct → sRGB → dither
async function processImagesWasm(inputData, refData, method, config, histogramMode, histogramDitherMode, outputDitherMode, colorAwareHistogram, histogramDistanceSpace, colorAwareOutput, outputDistanceSpace) {
    sendProgress('process', 'Decoding images (precise)...', 5);

    // First decode input to check for alpha
    const inputCheck = craWasm.load_image_wasm(new Uint8Array(inputData));
    const originalHasAlpha = inputCheck.has_alpha;

    // Histogram processing doesn't support alpha yet - discard alpha for now
    const inputHasAlpha = false;
    if (originalHasAlpha) {
        sendConsole('Note: Alpha channel discarded (histogram processing does not support alpha)');
    }

    // Decode with full precision (16-bit + ICC extraction)
    // Always use RGB buffer since histogram processing discards alpha
    const inputImg = decodeImagePrecise(inputData, false);
    const refImg = decodeImagePrecise(refData, false);  // Reference doesn't need alpha

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
    // Precise pipeline using BufferF32x4 (no intermediate copies)
    // decode → ICC transform (if non-sRGB) → linear → process → sRGB → dither
    // ========================================================================

    sendProgress('process', 'Processing color profiles...', 15);

    // Work with the decoded buffers
    let inputBuffer = inputImg.buffer;
    let refBuffer = refImg.buffer;

    let inputAlreadyLinear = false;
    let refAlreadyLinear = false;

    // Apply color space conversion
    // Priority: CICP sRGB/linear (authoritative) > ICC > CICP conversion > assume sRGB

    // Input image
    if (inputImg.isCicpSrgb) {
        sendConsole('  Input: CICP indicates sRGB');
        craWasm.srgb_to_linear_wasm(inputBuffer);
        inputAlreadyLinear = true;
    } else if (inputImg.isCicpLinear) {
        sendConsole('  Input: CICP indicates linear sRGB');
        inputAlreadyLinear = true;
    } else if (inputImg.iccProfile && inputImg.iccProfile.length > 0) {
        sendConsole('  Applying ICC transform to input...');
        craWasm.transform_icc_to_linear_srgb_wasm(inputBuffer, inputImg.width, inputImg.height, inputImg.iccProfile);
        inputAlreadyLinear = true;
    } else if (inputImg.isCicpNeedsConversion && inputImg.loadedImage) {
        sendConsole('  Applying CICP transform to input...');
        craWasm.transform_cicp_to_linear_srgb_wasm(inputBuffer, inputImg.width, inputImg.height, inputImg.loadedImage);
        inputAlreadyLinear = true;
    }

    // Reference image
    if (refImg.isCicpSrgb) {
        sendConsole('  Reference: CICP indicates sRGB');
        craWasm.srgb_to_linear_wasm(refBuffer);
        refAlreadyLinear = true;
    } else if (refImg.isCicpLinear) {
        sendConsole('  Reference: CICP indicates linear sRGB');
        refAlreadyLinear = true;
    } else if (refImg.iccProfile && refImg.iccProfile.length > 0) {
        sendConsole('  Applying ICC transform to reference...');
        craWasm.transform_icc_to_linear_srgb_wasm(refBuffer, refImg.width, refImg.height, refImg.iccProfile);
        refAlreadyLinear = true;
    } else if (refImg.isCicpNeedsConversion && refImg.loadedImage) {
        sendConsole('  Applying CICP transform to reference...');
        craWasm.transform_cicp_to_linear_srgb_wasm(refBuffer, refImg.width, refImg.height, refImg.loadedImage);
        refAlreadyLinear = true;
    }

    // Convert to linear if not already done
    sendProgress('process', 'Converting to linear RGB...', 25);
    if (!inputAlreadyLinear) {
        craWasm.srgb_to_linear_wasm(inputBuffer);
    }
    if (!refAlreadyLinear) {
        craWasm.srgb_to_linear_wasm(refBuffer);
    }

    // Un-premultiply alpha if needed (auto: only EXR has premultiplied alpha)
    // Must happen in linear space after color space conversion
    // Note: Skipped when alpha is discarded for histogram processing (inputHasAlpha is false)
    if (inputHasAlpha && inputImg.isFormatPremultipliedDefault) {
        sendConsole('  Un-premultiplying alpha (EXR detected)...');
        craWasm.unpremultiply_alpha_wasm(inputBuffer);
    }

    // Color correction (in linear RGB space)
    sendProgress('process', 'Running color correction...', 30);
    const resultBuffer = craWasm.color_correct_with_progress_wasm(
        inputBuffer,
        refBuffer,
        inputImg.width,
        inputImg.height,
        refImg.width,
        refImg.height,
        methodCode,
        luminosityFlag,
        histogramMode,
        histogramDitherMode,
        colorAwareHistogram,
        histogramDistanceSpace,
        (progress) => sendProgress('process', 'Running color correction...', 30 + Math.round(progress * 25))
    );

    // Convert linear back to sRGB (0-1) in-place
    sendProgress('process', 'Converting to sRGB...', 60);
    craWasm.linear_to_srgb_wasm(resultBuffer);

    // Denormalize to 0-255 in-place
    sendProgress('process', 'Denormalizing...', 65);
    craWasm.denormalize_clamped_wasm(resultBuffer);

    // Dither to 8-bit output
    sendProgress('process', 'Dithering output...', 70);
    const ditherTechnique = colorAwareOutput ? 2 : 1;

    let resultData;
    if (inputHasAlpha) {
        // Use RGBA dithering - alpha is passed through without dithering
        const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
            resultBuffer,
            inputImg.width,
            inputImg.height,
            8, 8, 8, 8,  // RGBA8888
            ditherTechnique,
            outputDitherMode,
            255,  // alpha_mode: use same as outputDitherMode
            outputDistanceSpace,
            0,  // seed
            (progress) => sendProgress('process', 'Dithering output...', 70 + Math.round(progress * 10))
        );
        resultData = ditheredBuffer.to_vec();
    } else {
        // Use RGB dithering for images without alpha
        const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
            resultBuffer,
            inputImg.width,
            inputImg.height,
            8, 8, 8,  // RGB888
            ditherTechnique,
            outputDitherMode,
            outputDistanceSpace,
            0,  // seed
            (progress) => sendProgress('process', 'Dithering output...', 70 + Math.round(progress * 10))
        );
        resultData = ditheredBuffer.to_vec();
    }

    const elapsed = performance.now() - startTime;
    sendConsole(`Processing completed in ${elapsed.toFixed(0)}ms`);

    sendProgress('process', 'Encoding result...', 80);
    const outputData = await encodePng(resultData, inputImg.width, inputImg.height, inputHasAlpha);

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
