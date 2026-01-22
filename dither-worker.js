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

// Supersample mode constants
const SUPERSAMPLE_NONE = 'none';
const SUPERSAMPLE_TENT_VOLUME = 'tent-volume';

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
            isPaletted = false,  // True for paletted mode (web-safe 216 colors)
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
            bitsGrayA = 0,  // Grayscale alpha bit depth: 0 to strip alpha, 1-8 to dither (LA format)
            mode,
            alphaMode = 255,  // Alpha dithering mode: 255 = use same as mode, otherwise separate mode for alpha
            isPerceptual,
            perceptualSpace,
            seed,
            tonemapping = 'none',  // 'none', 'aces', 'aces-inverse'
            supersample = 'none'   // 'none', 'tent-volume'
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
        // Tonemapping requires linear path (operates on linear values)
        // Note: isPaletted does NOT require linear path by itself (uses sRGB input like RGB dithering)
        const needsTonemapping = tonemapping !== 'none';
        const needsLinear = isGrayscale || doDownscale || loadedImage.has_non_srgb_icc || loadedImage.is_cicp_needs_conversion || inputIsLinear || needsUnpremultiply || needsTonemapping;

        const pixelCount = processWidth * processHeight;
        const outputData = new Uint8ClampedArray(pixelCount * 4);

        // For storing data for download
        let rgbInterleaved = null;
        let rgbaInterleaved = null;
        let grayChannel = null;
        let laInterleaved = null;

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

            // Step 2: Tent-volume expansion (if supersampling enabled)
            // Must happen before resize to match CLI order
            const useSupersampling = supersample === SUPERSAMPLE_TENT_VOLUME && tonemapping !== 'none';
            if (useSupersampling) {
                sendProgress(28, 'Expanding to tent-space...');
                const expanded = craWasm.tent_expand_wasm(buffer, currentWidth, currentHeight);
                buffer = expanded.buffer;
                currentWidth = expanded.width;
                currentHeight = expanded.height;
            }

            // Step 3: Rescale in linear space (if needed)
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
                // GRAYSCALE PATH: grayscale → tonemapping → tent_contract → sRGB → dither
                // (matches CLI grayscale path order)
                const preserveGrayAlpha = hasAlpha && bitsGrayA > 0;

                // Step 4a: Convert to grayscale (in tent-space if supersampling)
                sendProgress(40, 'Converting to grayscale...');
                let grayBuffer = craWasm.rgb_to_grayscale_wasm(buffer);

                // Step 5a: Apply tonemapping to grayscale (in tent-space if supersampling)
                if (tonemapping === 'aces') {
                    sendProgress(45, 'Applying tonemapping (ACES)...');
                    craWasm.gray_tonemap_aces_wasm(grayBuffer);
                } else if (tonemapping === 'aces-inverse') {
                    sendProgress(45, 'Applying tonemapping (ACES inverse)...');
                    craWasm.gray_tonemap_aces_inverse_wasm(grayBuffer);
                }

                // Step 6a: Tent-volume contraction (if supersampling enabled)
                // For grayscale, reconstruct as Pixel4, contract, extract back
                if (useSupersampling) {
                    sendProgress(50, 'Contracting from tent-space...');
                    // Reconstruct grayscale as RGB (R=G=B=L) for contraction
                    const grayAsRgb = craWasm.grayscale_to_rgb_wasm(grayBuffer, preserveGrayAlpha ? buffer : null);
                    const contracted = craWasm.tent_contract_wasm(grayAsRgb, currentWidth, currentHeight);
                    // Extract grayscale back from contracted RGB
                    grayBuffer = craWasm.rgb_to_grayscale_wasm(contracted.buffer);
                    if (preserveGrayAlpha) {
                        // Also contract alpha
                        buffer = contracted.buffer;
                    }
                    currentWidth = contracted.width;
                    currentHeight = contracted.height;
                }

                sendProgress(55, 'Applying gamma correction...');
                craWasm.gray_linear_to_srgb_wasm(grayBuffer);
                craWasm.gray_denormalize_wasm(grayBuffer);

                if (preserveGrayAlpha) {
                    // LA format: grayscale with alpha
                    // Extract alpha channel using WASM function (buffer is opaque, can't access directly from JS)
                    let alphaBuffer = craWasm.extract_alpha_wasm(buffer);
                    // Denormalize alpha from 0-1 to 0-255 range (gray_denormalize works on any BufferF32)
                    craWasm.gray_denormalize_wasm(alphaBuffer);

                    sendProgress(70, 'Dithering grayscale+alpha...');
                    const ditheredBuffer = craWasm.dither_la_with_progress_wasm(
                        grayBuffer, alphaBuffer, currentWidth, currentHeight, bitsGray, bitsGrayA, technique, mode, alphaMode, perceptualSpace, seed,
                        (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering grayscale+alpha...')
                    );
                    const laDithered = ditheredBuffer.to_vec();

                    // Store LA interleaved data for download (2 bytes per pixel: L, A)
                    laInterleaved = new Uint8Array(laDithered);

                    // Output as grayscale with alpha (R=G=B=L, A=A)
                    for (let i = 0; i < pixelCount; i++) {
                        const l = laDithered[i * 2];
                        const a = laDithered[i * 2 + 1];
                        outputData[i * 4] = l;
                        outputData[i * 4 + 1] = l;
                        outputData[i * 4 + 2] = l;
                        outputData[i * 4 + 3] = a;
                    }
                } else {
                    // Pure grayscale: no alpha
                    sendProgress(70, 'Dithering grayscale...');
                    const ditheredBuffer = craWasm.dither_gray_with_progress_wasm(
                        grayBuffer, currentWidth, currentHeight, bitsGray, technique, mode, perceptualSpace, seed,
                        (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering grayscale...')
                    );
                    const grayDithered = ditheredBuffer.to_vec();

                    // Store for download
                    grayChannel = new Uint8Array(grayDithered);

                    // Output as grayscale (R=G=B) - no alpha
                    for (let i = 0; i < pixelCount; i++) {
                        const v = grayDithered[i];
                        outputData[i * 4] = v;
                        outputData[i * 4 + 1] = v;
                        outputData[i * 4 + 2] = v;
                        outputData[i * 4 + 3] = 255;
                    }
                }
            } else if (isPaletted) {
                // PALETTED PATH: tonemapping → tent_contract → sRGB → paletted dither
                // Uses fixed palette (e.g., web-safe 216 colors) with integrated alpha distance

                // Step 4b: Apply tonemapping (in tent-space if supersampling)
                if (tonemapping === 'aces') {
                    sendProgress(45, 'Applying tonemapping (ACES)...');
                    craWasm.tonemap_aces_wasm(buffer);
                } else if (tonemapping === 'aces-inverse') {
                    sendProgress(45, 'Applying tonemapping (ACES inverse)...');
                    craWasm.tonemap_aces_inverse_wasm(buffer);
                }

                // Step 5b: Tent-volume contraction (if supersampling enabled)
                if (useSupersampling) {
                    sendProgress(50, 'Contracting from tent-space...');
                    const contracted = craWasm.tent_contract_wasm(buffer, currentWidth, currentHeight);
                    buffer = contracted.buffer;
                    currentWidth = contracted.width;
                    currentHeight = contracted.height;
                }

                sendProgress(55, 'Converting to sRGB...');
                craWasm.linear_to_srgb_wasm(buffer);

                sendProgress(60, 'Denormalizing...');
                craWasm.denormalize_clamped_wasm(buffer);

                sendProgress(70, 'Dithering with palette...');
                // palette_type: 0 = WebSafe (216 colors)
                const ditheredBuffer = craWasm.dither_paletted_with_progress_wasm(
                    buffer, currentWidth, currentHeight, 0, mode, perceptualSpace, seed,
                    (progress) => sendProgress(70 + Math.round(progress * 25), 'Dithering with palette...')
                );
                const ditheredData = ditheredBuffer.to_vec();

                // Paletted dithering always returns RGBA (4 bytes/pixel)
                rgbaInterleaved = new Uint8Array(ditheredData);

                // Copy RGBA directly to output
                for (let i = 0; i < pixelCount; i++) {
                    outputData[i * 4] = ditheredData[i * 4];
                    outputData[i * 4 + 1] = ditheredData[i * 4 + 1];
                    outputData[i * 4 + 2] = ditheredData[i * 4 + 2];
                    outputData[i * 4 + 3] = ditheredData[i * 4 + 3];
                }
            } else {
                // RGB PATH: tonemapping → tent_contract → sRGB → dither
                // (matches CLI RGB path order)

                // Step 4b: Apply tonemapping (in tent-space if supersampling)
                if (tonemapping === 'aces') {
                    sendProgress(45, 'Applying tonemapping (ACES)...');
                    craWasm.tonemap_aces_wasm(buffer);
                } else if (tonemapping === 'aces-inverse') {
                    sendProgress(45, 'Applying tonemapping (ACES inverse)...');
                    craWasm.tonemap_aces_inverse_wasm(buffer);
                }

                // Step 5b: Tent-volume contraction (if supersampling enabled)
                if (useSupersampling) {
                    sendProgress(50, 'Contracting from tent-space...');
                    const contracted = craWasm.tent_contract_wasm(buffer, currentWidth, currentHeight);
                    buffer = contracted.buffer;
                    currentWidth = contracted.width;
                    currentHeight = contracted.height;
                }

                sendProgress(55, 'Converting to sRGB...');
                craWasm.linear_to_srgb_wasm(buffer);

                sendProgress(60, 'Denormalizing...');
                craWasm.denormalize_clamped_wasm(buffer);

                sendProgress(70, 'Dithering RGB...');
                if (hasAlpha) {
                    // Use RGBA dithering with alpha-aware error propagation
                    // When bitsA=0, returns RGB (3 bytes/pixel); otherwise RGBA (4 bytes/pixel)
                    const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                        buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, bitsA, technique, mode, alphaMode, perceptualSpace, seed,
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

            if (isPaletted) {
                // PALETTED PATH (sRGB-direct): paletted dither
                sendProgress(50, 'Dithering with palette...');
                // palette_type: 0 = WebSafe (216 colors)
                const ditheredBuffer = craWasm.dither_paletted_with_progress_wasm(
                    buffer, currentWidth, currentHeight, 0, mode, perceptualSpace, seed,
                    (progress) => sendProgress(50 + Math.round(progress * 45), 'Dithering with palette...')
                );
                const ditheredData = ditheredBuffer.to_vec();

                // Paletted dithering always returns RGBA (4 bytes/pixel)
                rgbaInterleaved = new Uint8Array(ditheredData);

                // Copy RGBA directly to output
                for (let i = 0; i < pixelCount; i++) {
                    outputData[i * 4] = ditheredData[i * 4];
                    outputData[i * 4 + 1] = ditheredData[i * 4 + 1];
                    outputData[i * 4 + 2] = ditheredData[i * 4 + 2];
                    outputData[i * 4 + 3] = ditheredData[i * 4 + 3];
                }
            } else if (hasAlpha) {
                // Use RGBA dithering with alpha-aware error propagation
                // When bitsA=0, returns RGB (3 bytes/pixel); otherwise RGBA (4 bytes/pixel)
                sendProgress(50, 'Dithering RGB...');
                const ditheredBuffer = craWasm.dither_rgba_with_progress_wasm(
                    buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, bitsA, technique, mode, alphaMode, perceptualSpace, seed,
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
                sendProgress(50, 'Dithering RGB...');
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

        // Determine if output has alpha:
        // - Grayscale: alpha when LA format (bitsGrayA > 0)
        // - Paletted: always has alpha (RGBA output)
        // - RGB/RGBA: alpha only when bitsA > 0
        const outputHasAlpha = isGrayscale
            ? (hasAlpha && bitsGrayA > 0)  // LA format
            : isPaletted
                ? true  // Paletted always outputs RGBA
                : (hasAlpha && bitsA > 0);  // ARGB format

        // Send result back
        sendComplete({
            outputData: outputData,
            width: processWidth,
            height: processHeight,
            isGrayscale: isGrayscale,
            isPaletted: isPaletted,
            bitsR: isGrayscale ? bitsGray : bitsR,
            bitsG: isGrayscale ? bitsGray : bitsG,
            bitsB: isGrayscale ? bitsGray : bitsB,
            bitsA: isGrayscale ? (hasAlpha ? bitsGrayA : 0) : (isPaletted ? 8 : (hasAlpha ? bitsA : 0)),
            mode: mode,
            rgbInterleaved: rgbInterleaved,
            rgbaInterleaved: rgbaInterleaved,
            grayChannel: grayChannel,
            laInterleaved: laInterleaved,
            hasAlpha: outputHasAlpha  // True only if input had alpha AND we preserved it (or paletted)
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
