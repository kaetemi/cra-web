// Web Worker for resize processing
// Runs WASM rescaling in a separate thread to keep UI responsive
// Uses precise WASM decoding to preserve 16-bit precision

let wasmReady = false;
let craWasm = null;

// Current request ID for tracking concurrent requests
let currentRequestId = null;

// Send progress update to main thread
function sendProgress(percent) {
    self.postMessage({ type: 'progress', percent, requestId: currentRequestId });
}

// Send error to main thread
function sendError(error) {
    self.postMessage({ type: 'error', message: error.message || String(error), requestId: currentRequestId });
}

// Send completion to main thread
function sendComplete(outputData, width, height, hasAlpha = false) {
    self.postMessage({ type: 'complete', outputData, width, height, hasAlpha, requestId: currentRequestId });
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

// No separate decode functions needed - use LoadedImage pattern directly

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
            perceptualSpace,
            inputIsLinear  // True if input is already linear (normal maps, data textures)
        } = params;

        // Single load - keeps u8/u16 pixels, converts on demand (CLI pattern)
        const loadedImage = craWasm.load_image_wasm(new Uint8Array(fileBytes));
        const srcWidth = loadedImage.width;
        const srcHeight = loadedImage.height;
        const hasAlpha = loadedImage.has_alpha;

        // Check for premultiplied alpha (auto: only EXR has premultiplied alpha by default)
        const needsUnpremultiply = hasAlpha && loadedImage.is_format_premultiplied_default;

        // Resize always needs linear path - use RGBA to preserve alpha
        let buffer = loadedImage.to_normalized_buffer_rgba();

        // Convert to linear
        // Priority: CICP sRGB/linear (authoritative) > ICC > CICP conversion > assume sRGB
        // Alpha passes through unchanged
        if (loadedImage.is_cicp_srgb) {
            // CICP says sRGB - use builtin gamma decode
            craWasm.srgb_to_linear_wasm(buffer);
        } else if (loadedImage.is_cicp_linear) {
            // CICP says linear - no conversion needed
        } else if (loadedImage.has_non_srgb_icc) {
            // Non-sRGB ICC profile - use ICC transform
            const iccProfile = loadedImage.get_icc_profile();
            craWasm.transform_icc_to_linear_srgb_wasm(buffer, srcWidth, srcHeight, iccProfile);
        } else if (loadedImage.is_cicp_needs_conversion) {
            // CICP indicates non-sRGB (e.g., Display P3) but no ICC - use CICP transform
            craWasm.transform_cicp_to_linear_srgb_wasm(buffer, srcWidth, srcHeight, loadedImage);
        } else if (inputIsLinear) {
            // Input is already linear - no conversion needed
        } else {
            // Default: assume sRGB
            craWasm.srgb_to_linear_wasm(buffer);
        }

        // Un-premultiply alpha if needed (must be done in linear space)
        if (needsUnpremultiply) {
            craWasm.unpremultiply_alpha_wasm(buffer);
        }

        // Step 4: Rescale in linear space with progress (handles all 4 channels)
        const resizedBuffer = craWasm.rescale_rgb_with_progress_wasm(
            buffer,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            interpolation,
            scaleMode,
            (progress) => sendProgress(Math.round(progress * 80))  // 0-80% for resize
        );

        sendProgress(85);

        // Step 5: Convert linear back to sRGB (0-1) in-place (alpha passes through)
        craWasm.linear_to_srgb_wasm(resizedBuffer);

        // Step 6: Denormalize to 0-255 in-place (all 4 channels)
        craWasm.denormalize_clamped_wasm(resizedBuffer);

        sendProgress(90);

        // Step 7: Dither to 8-bit output
        let outputData;
        const dstPixels = dstWidth * dstHeight;

        if (hasAlpha) {
            // Use RGBA dithering with alpha-aware error propagation
            const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                resizedBuffer,
                dstWidth, dstHeight,
                8, 8, 8, 8,  // RGB at 8-bit, alpha at 8-bit
                ditherTechnique,
                ditherMode,
                perceptualSpace,
                0,
                (progress) => sendProgress(90 + Math.round(progress * 10))
            );
            // RGBA output is already in correct format for ImageData
            outputData = new Uint8ClampedArray(ditheredBuffer.to_vec());
        } else {
            // Use RGB dithering for images without alpha
            const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                resizedBuffer,
                dstWidth, dstHeight,
                8, 8, 8,
                ditherTechnique,
                ditherMode,
                perceptualSpace,
                0,
                (progress) => sendProgress(90 + Math.round(progress * 10))
            );
            const dithered = ditheredBuffer.to_vec();

            // Convert RGB to RGBA for ImageData (alpha = 255)
            outputData = new Uint8ClampedArray(dstPixels * 4);
            for (let i = 0; i < dstPixels; i++) {
                outputData[i * 4] = dithered[i * 3];
                outputData[i * 4 + 1] = dithered[i * 3 + 1];
                outputData[i * 4 + 2] = dithered[i * 3 + 2];
                outputData[i * 4 + 3] = 255;
            }
        }

        sendComplete(outputData, dstWidth, dstHeight, hasAlpha);

    } catch (error) {
        sendError(error);
    }
}

// Process sRGB (bad) resize request - for comparison purposes
// Note: Even for the "bad" comparison, we still handle ICC correctly to show
// only the difference between linear vs sRGB interpolation, not ICC handling
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
            perceptualSpace,
            inputIsLinear
        } = params;

        // Single load - keeps u8/u16 pixels, converts on demand
        const loadedImage = craWasm.load_image_wasm(new Uint8Array(fileBytes));
        const srcWidth = loadedImage.width;
        const srcHeight = loadedImage.height;
        const hasAlpha = loadedImage.has_alpha;

        // Check for premultiplied alpha (auto: only EXR has premultiplied alpha by default)
        const needsUnpremultiply = hasAlpha && loadedImage.is_format_premultiplied_default;

        // Use RGBA buffer to preserve alpha
        let buffer = loadedImage.to_normalized_buffer_rgba();

        // Handle color profile conversion for fair comparison
        // Even for "bad" sRGB resize, we need to convert to linear first if we need to un-premultiply
        // Priority: CICP sRGB/linear (authoritative) > ICC > CICP conversion > assume sRGB
        if (loadedImage.is_cicp_srgb) {
            // CICP says sRGB - just handle un-premultiply if needed
            if (needsUnpremultiply) {
                craWasm.srgb_to_linear_wasm(buffer);
                craWasm.unpremultiply_alpha_wasm(buffer);
                craWasm.linear_to_srgb_wasm(buffer);
            }
        } else if (loadedImage.is_cicp_linear) {
            // CICP says linear - un-premultiply if needed, then convert to sRGB
            if (needsUnpremultiply) {
                craWasm.unpremultiply_alpha_wasm(buffer);
            }
            craWasm.linear_to_srgb_wasm(buffer);
        } else if (loadedImage.has_non_srgb_icc) {
            // Non-sRGB ICC profile → linear sRGB → (un-premultiply if needed) → sRGB
            const iccProfile = loadedImage.get_icc_profile();
            craWasm.transform_icc_to_linear_srgb_wasm(buffer, srcWidth, srcHeight, iccProfile);
            if (needsUnpremultiply) {
                craWasm.unpremultiply_alpha_wasm(buffer);
            }
            craWasm.linear_to_srgb_wasm(buffer);
        } else if (loadedImage.is_cicp_needs_conversion) {
            // CICP indicates non-sRGB (e.g., Display P3) but no ICC - use CICP transform
            craWasm.transform_cicp_to_linear_srgb_wasm(buffer, srcWidth, srcHeight, loadedImage);
            if (needsUnpremultiply) {
                craWasm.unpremultiply_alpha_wasm(buffer);
            }
            craWasm.linear_to_srgb_wasm(buffer);
        } else if (inputIsLinear) {
            // Input is already linear - un-premultiply if needed, then convert to sRGB for the "bad" resize
            if (needsUnpremultiply) {
                craWasm.unpremultiply_alpha_wasm(buffer);
            }
            craWasm.linear_to_srgb_wasm(buffer);
        } else if (needsUnpremultiply) {
            // sRGB input but need to un-premultiply - must go to linear first
            craWasm.srgb_to_linear_wasm(buffer);
            craWasm.unpremultiply_alpha_wasm(buffer);
            craWasm.linear_to_srgb_wasm(buffer);
        }
        // else: already sRGB, no conversion or un-premultiply needed

        // WRONG - rescale directly in sRGB space (no linear conversion)
        // This demonstrates the incorrect result for comparison
        const resizedBuffer = craWasm.rescale_rgb_with_progress_wasm(
            buffer,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            interpolation,
            scaleMode,
            (progress) => sendProgress(Math.round(progress * 80))
        );

        // Step 4: Denormalize to 0-255 in-place
        craWasm.denormalize_clamped_wasm(resizedBuffer);

        sendProgress(90);

        // Step 5: Dither to 8-bit output
        let outputData;
        const dstPixels = dstWidth * dstHeight;

        if (hasAlpha) {
            // Use RGBA dithering with alpha-aware error propagation
            const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                resizedBuffer,
                dstWidth, dstHeight,
                8, 8, 8, 8,  // RGB at 8-bit, alpha at 8-bit
                ditherTechnique,
                ditherMode,
                perceptualSpace,
                0,
                (progress) => sendProgress(90 + Math.round(progress * 10))
            );
            outputData = new Uint8ClampedArray(ditheredBuffer.to_vec());
        } else {
            // Use RGB dithering for images without alpha
            const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                resizedBuffer,
                dstWidth, dstHeight,
                8, 8, 8,
                ditherTechnique,
                ditherMode,
                perceptualSpace,
                0,
                (progress) => sendProgress(90 + Math.round(progress * 10))
            );
            const dithered = ditheredBuffer.to_vec();

            // Convert RGB to RGBA for ImageData (alpha = 255)
            outputData = new Uint8ClampedArray(dstPixels * 4);
            for (let i = 0; i < dstPixels; i++) {
                outputData[i * 4] = dithered[i * 3];
                outputData[i * 4 + 1] = dithered[i * 3 + 1];
                outputData[i * 4 + 2] = dithered[i * 3 + 2];
                outputData[i * 4 + 3] = 255;
            }
        }

        sendComplete(outputData, dstWidth, dstHeight, hasAlpha);

    } catch (error) {
        sendError(error);
    }
}

// Process resize from raw RGBA pixels (from canvas, already u8 sRGB)
// Simpler path that bypasses file decoding - for use with browser-loaded images
function processResizePixels(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const { pixelData, srcWidth, srcHeight, dstWidth, dstHeight, interpolation, scaleMode } = params;

        // Convert RGBA u8 to RGBA f32 normalized (0-1)
        const pixelCount = srcWidth * srcHeight;
        const rgbaData = new Float32Array(pixelCount * 4);
        for (let i = 0; i < pixelCount; i++) {
            rgbaData[i * 4] = pixelData[i * 4] / 255.0;
            rgbaData[i * 4 + 1] = pixelData[i * 4 + 1] / 255.0;
            rgbaData[i * 4 + 2] = pixelData[i * 4 + 2] / 255.0;
            rgbaData[i * 4 + 3] = 0.0;  // alpha unused
        }

        // Create BufferF32x4 from RGBA f32 data
        let buffer = craWasm.create_buffer_from_rgba_wasm(Array.from(rgbaData), pixelCount);

        // Convert sRGB to linear
        craWasm.srgb_to_linear_wasm(buffer);

        sendProgress(20);

        // Rescale in linear space with progress
        const resizedBuffer = craWasm.rescale_rgb_with_progress_wasm(
            buffer,
            srcWidth, srcHeight,
            dstWidth, dstHeight,
            interpolation,
            scaleMode,
            (progress) => sendProgress(20 + Math.round(progress * 60))
        );

        sendProgress(85);

        // Convert linear back to sRGB in-place
        craWasm.linear_to_srgb_wasm(resizedBuffer);

        // Denormalize to 0-255 in-place
        craWasm.denormalize_clamped_wasm(resizedBuffer);

        sendProgress(90);

        // Dither to RGB888
        const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
            resizedBuffer,
            dstWidth, dstHeight,
            8, 8, 8,
            2,  // ColorAware
            4,  // Mixed
            1,  // OKLab
            0,  // seed
            (progress) => sendProgress(90 + Math.round(progress * 10))  // 90-100% for dither
        );

        // Extract final RGB u8 data
        const dithered = ditheredBuffer.to_vec();

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
        console.error('[resize-worker] Error:', error);
        sendError(error);
    }
}

// Handle messages from main thread
self.onmessage = function(e) {
    const { type, requestId, ...data } = e.data;
    currentRequestId = requestId || null;

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
        case 'resize-pixels':
            processResizePixels(data);
            break;
    }
};
