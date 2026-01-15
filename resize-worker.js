// Web Worker for resize processing
// Runs WASM rescaling in a separate thread to keep UI responsive

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

// Process resize request
function processResize(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const {
            imageData,  // Uint8ClampedArray RGBA
            srcWidth,
            srcHeight,
            dstWidth,
            dstHeight,
            interpolation,
            scaleMode,
            ditherMode,
            ditherTechnique,
            perceptualSpace
        } = params;

        // Convert RGBA to RGB interleaved Uint8Array
        const pixels = srcWidth * srcHeight;
        const rgbData = new Uint8Array(pixels * 3);
        for (let i = 0; i < pixels; i++) {
            rgbData[i * 3] = imageData[i * 4];
            rgbData[i * 3 + 1] = imageData[i * 4 + 1];
            rgbData[i * 3 + 2] = imageData[i * 4 + 2];
        }

        // Step 1: Unpack u8 to f32 (0-255)
        const srgbF32_255 = craWasm.unpack_u8_to_f32_wasm(rgbData, srcWidth, srcHeight);

        // Step 2: Normalize to 0-1
        const srgbF32 = craWasm.normalize_f32_wasm(srgbF32_255, srcWidth, srcHeight);

        // Step 3: Convert sRGB to linear
        const linearRgb = craWasm.srgb_to_linear_f32_wasm(srgbF32, srcWidth, srcHeight);

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

// Process sRGB (bad) resize request
function processSrgbResize(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const {
            imageData,
            srcWidth,
            srcHeight,
            dstWidth,
            dstHeight,
            interpolation,
            scaleMode,
            ditherMode,
            ditherTechnique,
            perceptualSpace
        } = params;

        // Convert RGBA to RGB interleaved Uint8Array
        const pixels = srcWidth * srcHeight;
        const rgbData = new Uint8Array(pixels * 3);
        for (let i = 0; i < pixels; i++) {
            rgbData[i * 3] = imageData[i * 4];
            rgbData[i * 3 + 1] = imageData[i * 4 + 1];
            rgbData[i * 3 + 2] = imageData[i * 4 + 2];
        }

        // Step 1: Unpack u8 to f32 (0-255)
        const srgbF32_255 = craWasm.unpack_u8_to_f32_wasm(rgbData, srcWidth, srcHeight);

        // Step 2: Normalize to 0-1 (still sRGB)
        const srgbF32 = craWasm.normalize_f32_wasm(srgbF32_255, srcWidth, srcHeight);

        // Step 3: WRONG - rescale directly in sRGB space (no linear conversion)
        const srgbResized = craWasm.rescale_linear_rgb_with_progress_wasm(
            srgbF32,
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
