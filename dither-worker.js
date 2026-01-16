// Web Worker for dither processing
// Runs WASM dithering in a separate thread to keep UI responsive

let wasmReady = false;
let craWasm = null;

// Send progress update to main thread
function sendProgress(percent, message) {
    self.postMessage({ type: 'progress', percent, message });
}

// Send error to main thread
function sendError(error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
}

// Send completion to main thread
function sendComplete(result) {
    self.postMessage({ type: 'complete', result });
}

// Initialize WASM
async function initialize() {
    try {
        sendProgress(0, 'Loading WASM module...');
        craWasm = await import('./wasm_cra/cra_wasm.js');
        await craWasm.default();
        wasmReady = true;
        self.postMessage({ type: 'ready' });
    } catch (error) {
        sendError(error);
    }
}

// Scale mode constants (must match Rust enum)
const SCALE_MODE_INDEPENDENT = 0;
const SCALE_MODE_UNIFORM_WIDTH = 1;
const SCALE_MODE_UNIFORM_HEIGHT = 2;

// Process dither request
// Uses single-load pattern following CLI's dual-path approach:
// - Linear path: load normalized (0-1) + ICC profile, process, then dither
// - sRGB direct path: load sRGB (0-255) directly, dither only (no linear conversion)
function processDither(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const {
            fileBytes,
            originalWidth,
            originalHeight,
            inputIsLinear,  // True if input is already linear (normal maps, data textures)
            isGrayscale,
            doDownscale,
            processWidth,
            processHeight,
            scaleMethod,
            primaryDimension,
            bitsR,
            bitsG,
            bitsB,
            bitsGray,
            mode,
            isPerceptual,
            perceptualSpace,
            seed
        } = params;

        sendProgress(5, 'Loading image...');

        // Single load - keeps u8/u16 pixels, converts on demand (CLI pattern)
        const loadedImage = craWasm.load_image_wasm(new Uint8Array(fileBytes));

        // Determine if linear processing is needed (matches CLI pattern)
        // Use loaded image's ICC check (single decode, no separate metadata call needed)
        // inputIsLinear counts as needing the linear path (already in linear, process there)
        const needsLinear = isGrayscale || doDownscale || loadedImage.has_non_srgb_icc || inputIsLinear;

        // Check if input has alpha channel (for output format decision)
        const hasAlpha = loadedImage.has_alpha;

        const pixelCount = processWidth * processHeight;
        const outputData = new Uint8ClampedArray(pixelCount * 4);

        // For storing data for download
        let rgbInterleaved = null;
        let grayChannel = null;

        const technique = isPerceptual ? 2 : 1;

        // Track current dimensions as we process
        let currentWidth = originalWidth;
        let currentHeight = originalHeight;

        if (needsLinear) {
            // Linear path: convert to normalized (0-1)
            // Always use RGBA buffer - Pixel4/BufferF32x4 is float4, no overhead
            // Images without alpha get alpha=1.0 automatically
            sendProgress(10, 'Converting to normalized format...');
            let buffer = loadedImage.to_normalized_buffer_rgba();

            sendProgress(20, 'Converting to linear color space...');
            // Step 1: Convert to linear (either via ICC, sRGB gamma, or already linear)
            if (loadedImage.has_non_srgb_icc) {
                const iccProfile = loadedImage.get_icc_profile();
                craWasm.transform_icc_to_linear_srgb_wasm(buffer, currentWidth, currentHeight, iccProfile);
            } else if (inputIsLinear) {
                // Input is already linear (normal maps, data textures) - no conversion needed
            } else {
                craWasm.srgb_to_linear_wasm(buffer);
            }

            // Step 2: Rescale in linear space (if needed)
            if (doDownscale) {
                sendProgress(30, 'Rescaling...');
                const scaleMode = primaryDimension === 'width'
                    ? SCALE_MODE_UNIFORM_WIDTH
                    : SCALE_MODE_UNIFORM_HEIGHT;
                buffer = craWasm.rescale_rgb_wasm(buffer, currentWidth, currentHeight, processWidth, processHeight, scaleMethod, scaleMode);
                currentWidth = processWidth;
                currentHeight = processHeight;
            }

            if (isGrayscale) {
                // Extract alpha before grayscale conversion (in 0-1 range, will be scaled to 0-255)
                const bufferData = buffer.as_slice ? buffer.as_slice() : buffer;

                sendProgress(50, 'Converting to grayscale...');
                let grayBuffer = craWasm.rgb_to_grayscale_wasm(buffer);

                sendProgress(60, 'Applying gamma correction...');
                craWasm.gray_linear_to_srgb_wasm(grayBuffer);
                craWasm.gray_denormalize_wasm(grayBuffer);

                sendProgress(70, 'Dithering grayscale...');
                const ditheredBuffer = craWasm.dither_gray_with_progress_wasm(
                    grayBuffer, currentWidth, currentHeight, bitsGray, technique, mode, perceptualSpace, seed,
                    (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering grayscale...')
                );
                const grayDithered = ditheredBuffer.to_vec();

                // Store for download
                grayChannel = new Uint8Array(grayDithered);

                // Output as grayscale (R=G=B) with alpha from buffer
                for (let i = 0; i < pixelCount; i++) {
                    const v = grayDithered[i];
                    outputData[i * 4] = v;
                    outputData[i * 4 + 1] = v;
                    outputData[i * 4 + 2] = v;
                    outputData[i * 4 + 3] = Math.round(bufferData[i * 4 + 3] * 255);
                }
            } else {
                sendProgress(50, 'Converting to sRGB...');
                craWasm.linear_to_srgb_wasm(buffer);

                sendProgress(60, 'Denormalizing...');
                craWasm.denormalize_clamped_wasm(buffer);

                // Get alpha from buffer before dithering (now in 0-255 range after denormalize)
                const bufferData = buffer.as_slice ? buffer.as_slice() : buffer;

                sendProgress(70, 'Dithering RGB...');
                const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                    buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, technique, mode, perceptualSpace, seed,
                    (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering RGB...')
                );
                const rgbDithered = ditheredBuffer.to_vec();

                // Keep interleaved data for binary export
                rgbInterleaved = new Uint8Array(rgbDithered);

                // Convert to RGBA for display with alpha from buffer
                for (let i = 0; i < pixelCount; i++) {
                    outputData[i * 4] = rgbDithered[i * 3];
                    outputData[i * 4 + 1] = rgbDithered[i * 3 + 1];
                    outputData[i * 4 + 2] = rgbDithered[i * 3 + 2];
                    outputData[i * 4 + 3] = Math.round(bufferData[i * 4 + 3]);
                }
            }
        } else {
            // sRGB-direct path: bypass 0-1 range and linear conversion entirely
            // Always use RGBA buffer - Pixel4/BufferF32x4 is float4, no overhead
            // Images without alpha get alpha=255.0 automatically
            sendProgress(10, 'Converting to sRGB 0-255...');
            let buffer = loadedImage.to_srgb_255_buffer_rgba();

            // Get alpha from buffer (already in 0-255 range)
            const bufferData = buffer.as_slice ? buffer.as_slice() : buffer;

            sendProgress(50, 'Dithering RGB...');
            const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, technique, mode, perceptualSpace, seed,
                (progress) => sendProgress(50 + Math.round(progress * 45), 'Dithering RGB...')
            );
            const rgbDithered = ditheredBuffer.to_vec();

            // Keep interleaved data for binary export
            rgbInterleaved = new Uint8Array(rgbDithered);

            // Convert to RGBA for display with alpha from buffer
            for (let i = 0; i < pixelCount; i++) {
                outputData[i * 4] = rgbDithered[i * 3];
                outputData[i * 4 + 1] = rgbDithered[i * 3 + 1];
                outputData[i * 4 + 2] = rgbDithered[i * 3 + 2];
                outputData[i * 4 + 3] = Math.round(bufferData[i * 4 + 3]);
            }
        }

        sendProgress(100, 'Complete');

        // Send result back
        sendComplete({
            outputData: outputData,
            width: processWidth,
            height: processHeight,
            isGrayscale: isGrayscale,
            bitsR: isGrayscale ? bitsGray : bitsR,
            bitsG: isGrayscale ? bitsGray : bitsG,
            bitsB: isGrayscale ? bitsGray : bitsB,
            mode: mode,
            rgbInterleaved: rgbInterleaved,
            grayChannel: grayChannel,
            hasAlpha: hasAlpha
        });

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
        case 'dither':
            processDither(data);
            break;
    }
};
