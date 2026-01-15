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

// Decode image to BufferF32x4 (Pixel4 0-1) for linear processing path
function decodeToNormalized(fileBytes) {
    const metadata = craWasm.decode_metadata_wasm(new Uint8Array(fileBytes));
    const buffer = craWasm.decode_image_wasm(new Uint8Array(fileBytes));
    return {
        width: metadata[0],
        height: metadata[1],
        hasIcc: metadata[2] > 0.5,
        is16bit: metadata[3] > 0.5,
        buffer: buffer
    };
}

// Decode image to BufferF32x4 (Pixel4 0-255) for sRGB-direct path
function decodeToSrgb255(fileBytes) {
    const metadata = craWasm.decode_metadata_wasm(new Uint8Array(fileBytes));
    const buffer = craWasm.decode_image_srgb_255_wasm(new Uint8Array(fileBytes));
    return {
        width: metadata[0],
        height: metadata[1],
        hasIcc: metadata[2] > 0.5,
        is16bit: metadata[3] > 0.5,
        buffer: buffer
    };
}

// Check if image has non-sRGB ICC profile
function checkNonSrgbIcc(fileBytes, hasIcc) {
    if (!hasIcc) return false;
    const iccProfile = craWasm.extract_icc_profile_wasm(new Uint8Array(fileBytes));
    return iccProfile.length > 0 && !craWasm.is_icc_profile_srgb_wasm(iccProfile);
}

// Get ICC profile
function getIccProfile(fileBytes) {
    return craWasm.extract_icc_profile_wasm(new Uint8Array(fileBytes));
}

// Scale mode constants (must match Rust enum)
const SCALE_MODE_INDEPENDENT = 0;
const SCALE_MODE_UNIFORM_WIDTH = 1;
const SCALE_MODE_UNIFORM_HEIGHT = 2;

// Process dither request
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
            inputHasNonSrgbIcc,
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

        sendProgress(5, 'Decoding image...');

        // Determine if linear processing is needed
        const needsLinear = isGrayscale || doDownscale || inputHasNonSrgbIcc;

        const pixelCount = processWidth * processHeight;
        const outputData = new Uint8ClampedArray(pixelCount * 4);

        // Track current dimensions as we process
        let currentWidth = originalWidth;
        let currentHeight = originalHeight;

        // For storing channel data for download
        let rChannel = null;
        let gChannel = null;
        let bChannel = null;
        let grayChannel = null;

        const technique = isPerceptual ? 2 : 1;

        if (needsLinear) {
            sendProgress(10, 'Decoding with linear processing...');
            const decoded = decodeToNormalized(fileBytes);
            let buffer = decoded.buffer;

            sendProgress(20, 'Converting to linear color space...');
            // Step 1: Convert to linear (either via ICC or sRGB gamma)
            if (inputHasNonSrgbIcc) {
                const iccProfile = getIccProfile(fileBytes);
                craWasm.transform_icc_to_linear_srgb_wasm(buffer, currentWidth, currentHeight, iccProfile);
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
                sendProgress(50, 'Converting to grayscale...');
                let grayBuffer = craWasm.rgb_to_grayscale_wasm(buffer);

                sendProgress(60, 'Applying gamma correction...');
                craWasm.gray_linear_to_srgb_wasm(grayBuffer);
                craWasm.gray_denormalize_wasm(grayBuffer);

                sendProgress(70, 'Dithering grayscale...');
                const ditheredBuffer = craWasm.dither_gray_wasm(grayBuffer, currentWidth, currentHeight, bitsGray, technique, mode, perceptualSpace, seed);
                const grayDithered = ditheredBuffer.to_vec();

                // Store for download
                grayChannel = new Uint8Array(grayDithered);

                // Output as grayscale (R=G=B)
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
                craWasm.denormalize_wasm(buffer);

                sendProgress(70, 'Dithering RGB...');
                const ditheredBuffer = craWasm.dither_rgb_wasm(buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, technique, mode, perceptualSpace, seed);
                const rgbDithered = craWasm.to_u8_rgb_wasm(ditheredBuffer);

                // Store channels for download
                rChannel = new Uint8Array(pixelCount);
                gChannel = new Uint8Array(pixelCount);
                bChannel = new Uint8Array(pixelCount);

                for (let i = 0; i < pixelCount; i++) {
                    rChannel[i] = rgbDithered[i * 3];
                    gChannel[i] = rgbDithered[i * 3 + 1];
                    bChannel[i] = rgbDithered[i * 3 + 2];
                    outputData[i * 4] = rgbDithered[i * 3];
                    outputData[i * 4 + 1] = rgbDithered[i * 3 + 1];
                    outputData[i * 4 + 2] = rgbDithered[i * 3 + 2];
                    outputData[i * 4 + 3] = 255;
                }
            }
        } else {
            // sRGB-direct path: decode directly to 0-255, no intermediate
            sendProgress(10, 'Decoding (sRGB direct)...');
            const decoded = decodeToSrgb255(fileBytes);
            let buffer = decoded.buffer;

            sendProgress(50, 'Dithering RGB...');
            const ditheredBuffer = craWasm.dither_rgb_wasm(buffer, currentWidth, currentHeight, bitsR, bitsG, bitsB, technique, mode, perceptualSpace, seed);
            const rgbDithered = craWasm.to_u8_rgb_wasm(ditheredBuffer);

            // Store channels for download
            rChannel = new Uint8Array(pixelCount);
            gChannel = new Uint8Array(pixelCount);
            bChannel = new Uint8Array(pixelCount);

            for (let i = 0; i < pixelCount; i++) {
                rChannel[i] = rgbDithered[i * 3];
                gChannel[i] = rgbDithered[i * 3 + 1];
                bChannel[i] = rgbDithered[i * 3 + 2];
                outputData[i * 4] = rgbDithered[i * 3];
                outputData[i * 4 + 1] = rgbDithered[i * 3 + 1];
                outputData[i * 4 + 2] = rgbDithered[i * 3 + 2];
                outputData[i * 4 + 3] = 255;
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
            rChannel: rChannel,
            gChannel: gChannel,
            bChannel: bChannel,
            grayChannel: grayChannel
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
