// Web Worker for resize processing
// Runs WASM rescaling in a separate thread to keep UI responsive
// Uses precise WASM decoding to preserve 16-bit precision

let wasmReady = false;
let craWasm = null;

// Send progress update to main thread
function sendProgress(percent) {
    self.postMessage({ type: 'progress', percent });
}

// Send error to main thread
function sendError(error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
}

// Send completion to main thread
function sendComplete(outputData, width, height) {
    self.postMessage({ type: 'complete', outputData, width, height });
}

// Initialize WASM
async function initialize() {
    try {
        sendProgress(0);
        craWasm = await import('./wasm_cra/cra_wasm.js');
        await craWasm.default();
        wasmReady = true;
        self.postMessage({ type: 'ready' });
    } catch (error) {
        sendError(error);
    }
}

// Decode image from raw file bytes using WASM (preserves 16-bit precision)
function decodeImagePrecise(fileBytes) {
    const result = craWasm.decode_image_wasm(new Uint8Array(fileBytes));
    const width = result[0];
    const height = result[1];
    const hasIcc = result[2] > 0.5;
    const is16bit = result[3] > 0.5;
    const pixels = result.slice(4); // Interleaved RGB f32 (0-1)
    return { width, height, hasIcc, is16bit, pixels };
}

// Process resize request (with precise WASM decoding)
function processResize(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const {
            fileBytes,  // Raw file bytes for precise decoding
            dstWidth,
            dstHeight,
            interpolation,
            scaleMode,
            ditherMode,
            ditherTechnique,
            perceptualSpace
        } = params;

        // Decode image with WASM (preserves 16-bit precision)
        const decoded = decodeImagePrecise(fileBytes);
        const srcWidth = decoded.width;
        const srcHeight = decoded.height;
        const srgbNorm = decoded.pixels; // f32 0-1 sRGB

        // Convert sRGB (0-1) to linear
        const linearRgb = craWasm.srgb_to_linear_f32_wasm(srgbNorm, srcWidth, srcHeight);

        // Step 4: Rescale in linear space with progress
        const linearResized = craWasm.rescale_linear_rgb_with_progress_wasm(
            linearRgb,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            interpolation,
            scaleMode,
            (progress) => sendProgress(Math.round(progress * 80))  // 0-80% for resize
        );

        sendProgress(85);

        // Step 5: Convert linear back to sRGB (0-1)
        const srgbResized = craWasm.linear_to_srgb_f32_wasm(linearResized, dstWidth, dstHeight);

        // Step 6: Denormalize to 0-255
        const srgbResized_255 = craWasm.denormalize_f32_wasm(srgbResized, dstWidth, dstHeight);

        sendProgress(90);

        // Step 7: Dither to RGB888
        const dithered = craWasm.dither_output_wasm(
            srgbResized_255,
            dstWidth, dstHeight,
            8, 8, 8,
            ditherTechnique,
            ditherMode,
            perceptualSpace,
            0
        );

        sendProgress(100);

        // Convert to RGBA for ImageData
        const dstPixels = dstWidth * dstHeight;
        const outputData = new Uint8ClampedArray(dstPixels * 4);
        for (let i = 0; i < dstPixels; i++) {
            outputData[i * 4] = dithered[i * 3];
            outputData[i * 4 + 1] = dithered[i * 3 + 1];
            outputData[i * 4 + 2] = dithered[i * 3 + 2];
            outputData[i * 4 + 3] = 255;
        }

        sendComplete(outputData, dstWidth, dstHeight);

    } catch (error) {
        sendError(error);
    }
}

// Process sRGB (bad) resize request - for comparison purposes
function processSrgbResize(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const {
            fileBytes,
            dstWidth,
            dstHeight,
            interpolation,
            scaleMode,
            ditherMode,
            ditherTechnique,
            perceptualSpace
        } = params;

        // Decode image with WASM (preserves 16-bit precision)
        const decoded = decodeImagePrecise(fileBytes);
        const srcWidth = decoded.width;
        const srcHeight = decoded.height;
        const srgbNorm = decoded.pixels; // f32 0-1 sRGB

        // WRONG - rescale directly in sRGB space (no linear conversion)
        // This demonstrates the incorrect result for comparison
        const srgbResized = craWasm.rescale_linear_rgb_with_progress_wasm(
            srgbNorm,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            interpolation,
            scaleMode,
            (progress) => sendProgress(Math.round(progress * 80))
        );

        // Step 4: Denormalize to 0-255
        const srgbResized_255 = craWasm.denormalize_f32_wasm(srgbResized, dstWidth, dstHeight);

        sendProgress(90);

        // Step 5: Dither to RGB888
        const dithered = craWasm.dither_output_wasm(
            srgbResized_255,
            dstWidth, dstHeight,
            8, 8, 8,
            ditherTechnique,
            ditherMode,
            perceptualSpace,
            0
        );

        sendProgress(100);

        // Convert to RGBA for ImageData
        const dstPixels = dstWidth * dstHeight;
        const outputData = new Uint8ClampedArray(dstPixels * 4);
        for (let i = 0; i < dstPixels; i++) {
            outputData[i * 4] = dithered[i * 3];
            outputData[i * 4 + 1] = dithered[i * 3 + 1];
            outputData[i * 4 + 2] = dithered[i * 3 + 2];
            outputData[i * 4 + 3] = 255;
        }

        sendComplete(outputData, dstWidth, dstHeight);

    } catch (error) {
        sendError(error);
    }
}

// Handle messages from main thread
self.onmessage = function(e) {
    const { type, ...data } = e.data;

    switch (type) {
        case 'init':
            initialize();
            break;
        case 'resize':
            processResize(data);
            break;
        case 'resize-srgb':
            processSrgbResize(data);
            break;
    }
};
