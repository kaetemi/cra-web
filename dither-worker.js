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
            bitsA = 8,  // Alpha bit depth: 0 to strip alpha, 1-8 to dither
            bitsGray,
            mode,
            isPerceptual,
            perceptualSpace,
            seed
        } = params;

        sendProgress(5, 'Loading image...');

        // Check for SFI (safetensors) format and use appropriate loader
        const fileData = new Uint8Array(fileBytes);
        const isSfi = craWasm.is_sfi_format_wasm(fileData);
        const loadedImage = isSfi
            ? craWasm.load_sfi_wasm(fileData)
            : craWasm.load_image_wasm(fileData);

        // Check if input has alpha channel (for output format decision)
        const hasAlpha = loadedImage.has_alpha;

        // Check if input has premultiplied alpha (auto-detect: only EXR by default)
        const needsUnpremultiply = hasAlpha && loadedImage.is_format_premultiplied_default;

        // Determine if linear processing is needed (matches CLI pattern)
        // Use loaded image's CICP and ICC checks (single decode, no separate metadata call needed)
        // Priority: CICP (authoritative) > ICC > assume sRGB
        // inputIsLinear counts as needing the linear path (already in linear, process there)
        // Premultiplied alpha also requires linear path (un-premultiply must happen in linear space)
        const needsLinear = isGrayscale || doDownscale || loadedImage.has_non_srgb_icc || loadedImage.is_cicp_needs_conversion || inputIsLinear || needsUnpremultiply;

        const pixelCount = processWidth * processHeight;
        const outputData = new Uint8ClampedArray(pixelCount * 4);

        // For storing data for download
        let rgbInterleaved = null;
        let rgbaInterleaved = null;
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
            // Step 1: Convert to linear
            // Priority: CICP sRGB/linear (authoritative) > ICC > CICP conversion > assume sRGB
            if (loadedImage.is_cicp_srgb) {
                // CICP says sRGB - use builtin gamma decode
                craWasm.srgb_to_linear_wasm(buffer);
            } else if (loadedImage.is_cicp_linear) {
                // CICP says linear - no conversion needed
            } else if (loadedImage.has_non_srgb_icc) {
                // Non-sRGB ICC profile - use ICC transform
                const iccProfile = loadedImage.get_icc_profile();
                craWasm.transform_icc_to_linear_srgb_wasm(buffer, currentWidth, currentHeight, iccProfile);
            } else if (loadedImage.is_cicp_needs_conversion) {
                // CICP indicates non-sRGB (e.g., Display P3) but no ICC - use CICP transform
                craWasm.transform_cicp_to_linear_srgb_wasm(buffer, currentWidth, currentHeight, loadedImage);
            } else if (inputIsLinear) {
                // Input is already linear (normal maps, data textures) - no conversion needed
            } else {
                // Default: assume sRGB
                craWasm.srgb_to_linear_wasm(buffer);
            }

            // Step 1.5: Un-premultiply alpha if needed (must be done in linear space)
            if (needsUnpremultiply) {
                sendProgress(25, 'Un-premultiplying alpha...');
                craWasm.unpremultiply_alpha_wasm(buffer);
            }

            // Step 2: Rescale in linear space (if needed)
            // Use alpha-aware rescaling when image has alpha to prevent transparent pixels
            // from bleeding their color into opaque regions
            if (doDownscale) {
                sendProgress(30, 'Rescaling...');
                const scaleMode = primaryDimension === 'width'
                    ? SCALE_MODE_UNIFORM_WIDTH
                    : SCALE_MODE_UNIFORM_HEIGHT;
                buffer = hasAlpha
                    ? craWasm.rescale_rgb_alpha_wasm(buffer, currentWidth, currentHeight, processWidth, processHeight, scaleMethod, scaleMode)
                    : craWasm.rescale_rgb_wasm(buffer, currentWidth, currentHeight, processWidth, processHeight, scaleMethod, scaleMode);
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

                // Output as grayscale (R=G=B) - no alpha support for grayscale
                for (let i = 0; i < pixelCount; i++) {
                    const v = grayDithered[i];
                    outputData[i * 4] = v;
                    outputData[i * 4 + 1] = v;
                    outputData[i * 4 + 2] = v;
                    outputData[i * 4 + 3] = 255;
                }
            } else {
                sendProgress(50, 'Converting to sRGB...');
                craWasm.linear_to_srgb_wasm(buffer);

                sendProgress(60, 'Denormalizing...');
                craWasm.denormalize_clamped_wasm(buffer);

                sendProgress(70, 'Dithering RGB...');
                if (hasAlpha) {
                    // Use RGBA dithering with alpha-aware error propagation
                    // When bitsA=0, returns RGB (3 bytes/pixel); otherwise RGBA (4 bytes/pixel)
                    const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                        buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, bitsA, technique, mode, perceptualSpace, seed,
                        (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering...')
                    );
                    const ditheredData = ditheredBuffer.to_vec();

                    if (bitsA > 0) {
                        // Keep interleaved RGBA data for binary export
                        rgbaInterleaved = new Uint8Array(ditheredData);

                        // Copy RGBA directly to output
                        for (let i = 0; i < pixelCount; i++) {
                            outputData[i * 4] = ditheredData[i * 4];
                            outputData[i * 4 + 1] = ditheredData[i * 4 + 1];
                            outputData[i * 4 + 2] = ditheredData[i * 4 + 2];
                            outputData[i * 4 + 3] = ditheredData[i * 4 + 3];
                        }
                    } else {
                        // bitsA=0: alpha stripped, result is RGB (3 bytes/pixel)
                        rgbInterleaved = new Uint8Array(ditheredData);

                        // Convert RGB to RGBA for display (alpha = 255)
                        for (let i = 0; i < pixelCount; i++) {
                            outputData[i * 4] = ditheredData[i * 3];
                            outputData[i * 4 + 1] = ditheredData[i * 3 + 1];
                            outputData[i * 4 + 2] = ditheredData[i * 3 + 2];
                            outputData[i * 4 + 3] = 255;
                        }
                    }
                } else {
                    // Use RGB dithering for images without alpha
                    const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                        buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, technique, mode, perceptualSpace, seed,
                        (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering RGB...')
                    );
                    const rgbDithered = ditheredBuffer.to_vec();

                    // Keep interleaved RGB data for binary export
                    rgbInterleaved = new Uint8Array(rgbDithered);

                    // Convert to RGBA for display (alpha = 255 for opaque)
                    for (let i = 0; i < pixelCount; i++) {
                        outputData[i * 4] = rgbDithered[i * 3];
                        outputData[i * 4 + 1] = rgbDithered[i * 3 + 1];
                        outputData[i * 4 + 2] = rgbDithered[i * 3 + 2];
                        outputData[i * 4 + 3] = 255;
                    }
                }
            }
        } else {
            // sRGB-direct path: bypass 0-1 range and linear conversion entirely
            // Always use RGBA buffer - Pixel4/BufferF32x4 is float4, no overhead
            // Images without alpha get alpha=255.0 automatically
            sendProgress(10, 'Converting to sRGB 0-255...');
            let buffer = loadedImage.to_srgb_255_buffer_rgba();

            sendProgress(50, 'Dithering RGB...');
            if (hasAlpha) {
                // Use RGBA dithering with alpha-aware error propagation
                // When bitsA=0, returns RGB (3 bytes/pixel); otherwise RGBA (4 bytes/pixel)
                const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                    buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, bitsA, technique, mode, perceptualSpace, seed,
                    (progress) => sendProgress(50 + Math.round(progress * 45), 'Dithering...')
                );
                const ditheredData = ditheredBuffer.to_vec();

                if (bitsA > 0) {
                    // Keep interleaved RGBA data for binary export
                    rgbaInterleaved = new Uint8Array(ditheredData);

                    // Copy RGBA directly to output
                    for (let i = 0; i < pixelCount; i++) {
                        outputData[i * 4] = ditheredData[i * 4];
                        outputData[i * 4 + 1] = ditheredData[i * 4 + 1];
                        outputData[i * 4 + 2] = ditheredData[i * 4 + 2];
                        outputData[i * 4 + 3] = ditheredData[i * 4 + 3];
                    }
                } else {
                    // bitsA=0: alpha stripped, result is RGB (3 bytes/pixel)
                    rgbInterleaved = new Uint8Array(ditheredData);

                    // Convert RGB to RGBA for display (alpha = 255)
                    for (let i = 0; i < pixelCount; i++) {
                        outputData[i * 4] = ditheredData[i * 3];
                        outputData[i * 4 + 1] = ditheredData[i * 3 + 1];
                        outputData[i * 4 + 2] = ditheredData[i * 3 + 2];
                        outputData[i * 4 + 3] = 255;
                    }
                }
            } else {
                // Use RGB dithering for images without alpha
                const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                    buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, technique, mode, perceptualSpace, seed,
                    (progress) => sendProgress(50 + Math.round(progress * 45), 'Dithering RGB...')
                );
                const rgbDithered = ditheredBuffer.to_vec();

                // Keep interleaved RGB data for binary export
                rgbInterleaved = new Uint8Array(rgbDithered);

                // Convert to RGBA for display (alpha = 255 for opaque)
                for (let i = 0; i < pixelCount; i++) {
                    outputData[i * 4] = rgbDithered[i * 3];
                    outputData[i * 4 + 1] = rgbDithered[i * 3 + 1];
                    outputData[i * 4 + 2] = rgbDithered[i * 3 + 2];
                    outputData[i * 4 + 3] = 255;
                }
            }
        }

        sendProgress(100, 'Complete');

        // Grayscale has no alpha support; RGB has alpha only when bitsA > 0
        const outputHasAlpha = !isGrayscale && hasAlpha && bitsA > 0;

        // Send result back
        sendComplete({
            outputData: outputData,
            width: processWidth,
            height: processHeight,
            isGrayscale: isGrayscale,
            bitsR: isGrayscale ? bitsGray : bitsR,
            bitsG: isGrayscale ? bitsGray : bitsG,
            bitsB: isGrayscale ? bitsGray : bitsB,
            bitsA: isGrayscale ? 0 : (hasAlpha ? bitsA : 0),
            mode: mode,
            rgbInterleaved: rgbInterleaved,
            rgbaInterleaved: rgbaInterleaved,
            grayChannel: grayChannel,
            hasAlpha: outputHasAlpha  // True only if input had alpha AND we preserved it (bitsA > 0)
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
