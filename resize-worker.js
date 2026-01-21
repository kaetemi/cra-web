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

// Supersample mode constants
const SUPERSAMPLE_NONE = 0;
const SUPERSAMPLE_TENT_VOLUME = 1;
const SUPERSAMPLE_TENT_VOLUME_PRESCALE = 2;
const SUPERSAMPLE_TENT_LANCZOS3 = 3;
const SUPERSAMPLE_TENT_LANCZOS3_PRESCALE = 4;

// Supersample mode helpers
const isVolumeMode = (mode) => mode === SUPERSAMPLE_TENT_VOLUME || mode === SUPERSAMPLE_TENT_VOLUME_PRESCALE;
const isLanczos3Mode = (mode) => mode === SUPERSAMPLE_TENT_LANCZOS3 || mode === SUPERSAMPLE_TENT_LANCZOS3_PRESCALE;
const isExplicitMode = (mode) => mode === SUPERSAMPLE_TENT_VOLUME || mode === SUPERSAMPLE_TENT_LANCZOS3;
const isPrescaleMode = (mode) => mode === SUPERSAMPLE_TENT_VOLUME_PRESCALE || mode === SUPERSAMPLE_TENT_LANCZOS3_PRESCALE;
const isSupersampling = (mode) => mode !== SUPERSAMPLE_NONE;

// TentMode constants for WASM
const TENT_MODE_OFF = 0;
const TENT_MODE_SAMPLE_TO_SAMPLE = 1;
const TENT_MODE_PRESCALE = 2;

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
            inputIsLinear,  // True if input is already linear (normal maps, data textures)
            supersample = SUPERSAMPLE_NONE,  // Supersampling mode
            tonemapping = 'none'  // 'none', 'aces', 'aces-inverse'
        } = params;

        // Check for SFI (safetensors) format and use appropriate loader
        const fileData = new Uint8Array(fileBytes);
        const isSfi = craWasm.is_sfi_format_wasm(fileData);
        const loadedImage = isSfi
            ? craWasm.load_sfi_wasm(fileData)
            : craWasm.load_image_wasm(fileData);
        let srcWidth = loadedImage.width;
        let srcHeight = loadedImage.height;
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
        // Use alpha-aware rescaling for images with alpha to prevent transparent pixels
        // from bleeding their color into opaque regions
        let resizedBuffer;
        let finalWidth = dstWidth;
        let finalHeight = dstHeight;

        if (isExplicitMode(supersample)) {
            // Explicit supersampling (Volume or Lanczos3):
            // 1. Expand to tent-space (2W+1, 2H+1)
            // 2. Resize to tent-space target (2*dst+1) with tent_mode=SampleToSample
            // 3. Contract back to box-space

            // Expand to tent-space (volume or lanczos3)
            const expandFn = isLanczos3Mode(supersample)
                ? craWasm.tent_expand_lanczos_wasm
                : craWasm.tent_expand_wasm;
            const expandResult = expandFn(buffer, srcWidth, srcHeight);
            buffer = expandResult.buffer;
            srcWidth = expandResult.width;
            srcHeight = expandResult.height;

            // Calculate tent-space target dimensions
            const tentTargetDims = craWasm.supersample_target_dimensions_wasm(dstWidth, dstHeight);
            const tentDstWidth = tentTargetDims[0];
            const tentDstHeight = tentTargetDims[1];

            // Resize in tent-space with tent_mode=SampleToSample
            const tentResized = hasAlpha
                ? craWasm.rescale_rgb_alpha_tent_with_progress_wasm(
                    buffer,
                    srcWidth, srcHeight,
                    tentDstWidth, tentDstHeight,
                    interpolation,
                    scaleMode,
                    TENT_MODE_SAMPLE_TO_SAMPLE,
                    (progress) => sendProgress(Math.round(progress * 70))
                )
                : craWasm.rescale_rgb_tent_with_progress_wasm(
                    buffer,
                    srcWidth, srcHeight,
                    tentDstWidth, tentDstHeight,
                    interpolation,
                    scaleMode,
                    TENT_MODE_SAMPLE_TO_SAMPLE,
                    (progress) => sendProgress(Math.round(progress * 70))
                );

            sendProgress(72);

            // Apply tonemapping in tent-space (before contraction)
            if (tonemapping === 'aces') {
                sendProgress(74, 'Applying tonemapping (ACES)...');
                craWasm.tonemap_aces_wasm(tentResized);
            } else if (tonemapping === 'aces-inverse') {
                sendProgress(74, 'Applying tonemapping (ACES inverse)...');
                craWasm.tonemap_aces_inverse_wasm(tentResized);
            }

            sendProgress(76);

            // Contract back to box-space (volume or lanczos3)
            const contractFn = isLanczos3Mode(supersample)
                ? craWasm.tent_contract_lanczos_wasm
                : craWasm.tent_contract_wasm;
            const contractResult = contractFn(tentResized, tentDstWidth, tentDstHeight);
            resizedBuffer = contractResult.buffer;
            finalWidth = contractResult.width;
            finalHeight = contractResult.height;

            sendProgress(80);
        } else if (isPrescaleMode(supersample)) {
            // Prescale supersampling (Volume or Lanczos3):
            // 1. Expand to tent-space (2W+1, 2H+1)
            // 2. Resize directly to final box-space with tent_mode=Prescale
            //    (the rescale integrates the contract step via coordinate mapping)

            // Store original dimensions for default behavior
            const originalWidth = srcWidth;
            const originalHeight = srcHeight;

            // Expand to tent-space (volume or lanczos3)
            const expandFn = isLanczos3Mode(supersample)
                ? craWasm.tent_expand_lanczos_wasm
                : craWasm.tent_expand_wasm;
            const expandResult = expandFn(buffer, srcWidth, srcHeight);
            buffer = expandResult.buffer;
            srcWidth = expandResult.width;
            srcHeight = expandResult.height;

            // Default interpolation: box (20) for volume, tent-lanczos3 (32) for lanczos3
            const defaultInterpolation = isLanczos3Mode(supersample) ? 32 : 20;
            const effectiveInterpolation = (dstWidth === originalWidth && dstHeight === originalHeight)
                ? defaultInterpolation
                : interpolation;

            // Resize directly from tent-space to final box-space with Prescale mode
            resizedBuffer = hasAlpha
                ? craWasm.rescale_rgb_alpha_tent_with_progress_wasm(
                    buffer,
                    srcWidth, srcHeight,
                    dstWidth, dstHeight,
                    effectiveInterpolation,
                    scaleMode,
                    TENT_MODE_PRESCALE,
                    (progress) => sendProgress(Math.round(progress * 75))
                )
                : craWasm.rescale_rgb_tent_with_progress_wasm(
                    buffer,
                    srcWidth, srcHeight,
                    dstWidth, dstHeight,
                    effectiveInterpolation,
                    scaleMode,
                    TENT_MODE_PRESCALE,
                    (progress) => sendProgress(Math.round(progress * 75))
                );

            sendProgress(78);

            // Apply tonemapping after prescale resize
            if (tonemapping === 'aces') {
                sendProgress(79, 'Applying tonemapping (ACES)...');
                craWasm.tonemap_aces_wasm(resizedBuffer);
            } else if (tonemapping === 'aces-inverse') {
                sendProgress(79, 'Applying tonemapping (ACES inverse)...');
                craWasm.tonemap_aces_inverse_wasm(resizedBuffer);
            }

            sendProgress(80);
        } else {
            // Standard resize without supersampling
            resizedBuffer = hasAlpha
                ? craWasm.rescale_rgb_alpha_with_progress_wasm(
                    buffer,
                    srcWidth, srcHeight,
                    dstWidth, dstHeight,
                    interpolation,
                    scaleMode,
                    (progress) => sendProgress(Math.round(progress * 80))  // 0-80% for resize
                )
                : craWasm.rescale_rgb_with_progress_wasm(
                    buffer,
                    srcWidth, srcHeight,
                    dstWidth, dstHeight,
                    interpolation,
                    scaleMode,
                    (progress) => sendProgress(Math.round(progress * 80))  // 0-80% for resize
                );
        }

        sendProgress(82);

        // Step 5: Apply tonemapping (if enabled and not supersampling - supersampling modes handle it in their pipelines)
        if (!isSupersampling(supersample)) {
            if (tonemapping === 'aces') {
                sendProgress(84, 'Applying tonemapping (ACES)...');
                craWasm.tonemap_aces_wasm(resizedBuffer);
            } else if (tonemapping === 'aces-inverse') {
                sendProgress(84, 'Applying tonemapping (ACES inverse)...');
                craWasm.tonemap_aces_inverse_wasm(resizedBuffer);
            }
        }

        sendProgress(86);

        // Step 6: Convert linear back to sRGB (0-1) in-place (alpha passes through)
        craWasm.linear_to_srgb_wasm(resizedBuffer);

        // Step 7: Denormalize to 0-255 in-place (all 4 channels)
        craWasm.denormalize_clamped_wasm(resizedBuffer);

        sendProgress(90);

        // Step 8: Dither to 8-bit output
        let outputData;
        const outputPixels = finalWidth * finalHeight;

        if (hasAlpha) {
            // Use RGBA dithering with alpha-aware error propagation
            const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                resizedBuffer,
                finalWidth, finalHeight,
                8, 8, 8, 8,  // RGB at 8-bit, alpha at 8-bit
                ditherTechnique,
                ditherMode,
                255,  // alpha_mode: use same as ditherMode
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
                finalWidth, finalHeight,
                8, 8, 8,
                ditherTechnique,
                ditherMode,
                perceptualSpace,
                0,
                (progress) => sendProgress(90 + Math.round(progress * 10))
            );
            const dithered = ditheredBuffer.to_vec();

            // Convert RGB to RGBA for ImageData (alpha = 255)
            outputData = new Uint8ClampedArray(outputPixels * 4);
            for (let i = 0; i < outputPixels; i++) {
                outputData[i * 4] = dithered[i * 3];
                outputData[i * 4 + 1] = dithered[i * 3 + 1];
                outputData[i * 4 + 2] = dithered[i * 3 + 2];
                outputData[i * 4 + 3] = 255;
            }
        }

        sendComplete(outputData, finalWidth, finalHeight, hasAlpha);

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

        // Check for SFI (safetensors) format and use appropriate loader
        const fileData = new Uint8Array(fileBytes);
        const isSfi = craWasm.is_sfi_format_wasm(fileData);
        const loadedImage = isSfi
            ? craWasm.load_sfi_wasm(fileData)
            : craWasm.load_image_wasm(fileData);
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
        // Still use alpha-aware rescaling if image has alpha (the comparison is about
        // linear vs sRGB interpolation, not alpha handling)
        const resizedBuffer = hasAlpha
            ? craWasm.rescale_rgb_alpha_with_progress_wasm(
                buffer,
                srcWidth, srcHeight,
                dstWidth, dstHeight,
                interpolation,
                scaleMode,
                (progress) => sendProgress(Math.round(progress * 80))
            )
            : craWasm.rescale_rgb_with_progress_wasm(
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
                255,  // alpha_mode: use same as ditherMode
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
        // Also check if image has any non-opaque alpha for alpha-aware rescaling
        const pixelCount = srcWidth * srcHeight;
        const rgbaData = new Float32Array(pixelCount * 4);
        let hasAlpha = false;
        for (let i = 0; i < pixelCount; i++) {
            rgbaData[i * 4] = pixelData[i * 4] / 255.0;
            rgbaData[i * 4 + 1] = pixelData[i * 4 + 1] / 255.0;
            rgbaData[i * 4 + 2] = pixelData[i * 4 + 2] / 255.0;
            rgbaData[i * 4 + 3] = pixelData[i * 4 + 3] / 255.0;
            if (pixelData[i * 4 + 3] < 255) {
                hasAlpha = true;
            }
        }

        // Create BufferF32x4 from RGBA f32 data
        let buffer = craWasm.create_buffer_from_rgba_wasm(Array.from(rgbaData), pixelCount);

        // Convert sRGB to linear (alpha passes through unchanged)
        craWasm.srgb_to_linear_wasm(buffer);

        sendProgress(20);

        // Rescale in linear space with progress
        // Use alpha-aware rescaling if image has any non-opaque pixels
        const resizedBuffer = hasAlpha
            ? craWasm.rescale_rgb_alpha_with_progress_wasm(
                buffer,
                srcWidth, srcHeight,
                dstWidth, dstHeight,
                interpolation,
                scaleMode,
                (progress) => sendProgress(20 + Math.round(progress * 60))
            )
            : craWasm.rescale_rgb_with_progress_wasm(
                buffer,
                srcWidth, srcHeight,
                dstWidth, dstHeight,
                interpolation,
                scaleMode,
                (progress) => sendProgress(20 + Math.round(progress * 60))
            );

        sendProgress(85);

        // Convert linear back to sRGB in-place (alpha passes through)
        craWasm.linear_to_srgb_wasm(resizedBuffer);

        // Denormalize to 0-255 in-place
        craWasm.denormalize_clamped_wasm(resizedBuffer);

        sendProgress(90);

        // Dither to output
        const dstPixels = dstWidth * dstHeight;
        let outputData;

        if (hasAlpha) {
            // Use RGBA dithering with alpha-aware error propagation
            const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                resizedBuffer,
                dstWidth, dstHeight,
                8, 8, 8, 8,  // RGB and alpha at 8-bit
                2,  // ColorAware
                4,  // Mixed
                255,  // alpha_mode: use same as ditherMode
                1,  // OKLab
                0,  // seed
                (progress) => sendProgress(90 + Math.round(progress * 10))
            );
            outputData = new Uint8ClampedArray(ditheredBuffer.to_vec());
        } else {
            // Use RGB dithering for fully opaque images
            const ditheredBuffer = craWasm.dither_rgb_with_progress_wasm(
                resizedBuffer,
                dstWidth, dstHeight,
                8, 8, 8,
                2,  // ColorAware
                4,  // Mixed
                1,  // OKLab
                0,  // seed
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
        console.error('[resize-worker] Error:', error);
        sendError(error);
    }
}

// Encode RGBA data to PNG (RGB or RGBA based on hasAlpha flag)
function encodePng(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const { rgbaData, width, height, hasAlpha } = params;

        let pngBytes;
        if (hasAlpha) {
            // Encode as RGBA PNG
            pngBytes = craWasm.encode_png_rgba_wasm(new Uint8Array(rgbaData), width, height);
        } else {
            // Extract RGB from RGBA and encode as RGB PNG
            const pixelCount = width * height;
            const rgbData = new Uint8Array(pixelCount * 3);
            for (let i = 0; i < pixelCount; i++) {
                rgbData[i * 3] = rgbaData[i * 4];
                rgbData[i * 3 + 1] = rgbaData[i * 4 + 1];
                rgbData[i * 3 + 2] = rgbaData[i * 4 + 2];
            }
            pngBytes = craWasm.encode_png_rgb_wasm(rgbData, width, height);
        }

        self.postMessage({
            type: 'png-encoded',
            pngBytes: pngBytes,
            requestId: currentRequestId
        });
    } catch (error) {
        sendError(error);
    }
}

// Get image dimensions without full processing (for EXR preview)
function getImageDimensions(params) {
    if (!wasmReady) {
        sendError(new Error('WASM not ready'));
        return;
    }

    try {
        const { fileBytes } = params;
        const fileData = new Uint8Array(fileBytes);
        const isSfi = craWasm.is_sfi_format_wasm(fileData);
        const loadedImage = isSfi
            ? craWasm.load_sfi_wasm(fileData)
            : craWasm.load_image_wasm(fileData);

        self.postMessage({
            type: 'dimensions',
            width: loadedImage.width,
            height: loadedImage.height,
            requestId: currentRequestId
        });
    } catch (error) {
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
        case 'encode-png':
            encodePng(data);
            break;
        case 'get-dimensions':
            getImageDimensions(data);
            break;
    }
};
