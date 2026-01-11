let wasm;

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let WASM_VECTOR_LEN = 0;

/**
 * Basic LAB histogram matching (WASM export)
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     keep_luminosity: If true, preserve original L channel
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} keep_luminosity
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_basic_lab(input_data, input_width, input_height, ref_data, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_basic_lab(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Basic Oklab histogram matching (WASM export)
 *
 * Oklab is a perceptually uniform color space with better hue linearity than LAB.
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     keep_luminosity: If true, preserve original L channel
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} keep_luminosity
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_basic_oklab(input_data, input_width, input_height, ref_data, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_basic_oklab(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Basic RGB histogram matching (WASM export)
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_basic_rgb(input_data, input_width, input_height, ref_data, ref_width, ref_height, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_basic_rgb(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * CRA LAB color correction (WASM export)
 *
 * Chroma Rotation Averaging in LAB color space. Rotates the AB chroma plane
 * at multiple angles, performs histogram matching at each rotation, then
 * averages the results.
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     keep_luminosity: If true, preserve original L channel
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} keep_luminosity
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_cra_lab(input_data, input_width, input_height, ref_data, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_cra_lab(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * CRA Oklab color correction (WASM export)
 *
 * Chroma Rotation Averaging in Oklab color space. Rotates the AB chroma plane
 * at multiple angles, performs histogram matching at each rotation, then
 * averages the results. Oklab provides better perceptual uniformity than LAB.
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     keep_luminosity: If true, preserve original L channel
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} keep_luminosity
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_cra_oklab(input_data, input_width, input_height, ref_data, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_cra_oklab(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, keep_luminosity, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * CRA RGB color correction (WASM export)
 *
 * Chroma Rotation Averaging in RGB space. Rotates the RGB cube around the
 * neutral gray axis (1,1,1) using Rodrigues' rotation formula.
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     use_perceptual: If true, use perceptual weighting
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} use_perceptual
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_cra_rgb(input_data, input_width, input_height, ref_data, ref_width, ref_height, use_perceptual, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_cra_rgb(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, use_perceptual, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Tiled CRA LAB color correction (WASM export)
 *
 * CRA with overlapping tile-based processing. Divides the image into blocks
 * with 50% overlap, applies CRA to each block, then blends results using
 * Hamming windows.
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     tiled_luminosity: If true, process L channel per-tile before global match
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} tiled_luminosity
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_tiled_lab(input_data, input_width, input_height, ref_data, ref_width, ref_height, tiled_luminosity, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_tiled_lab(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, tiled_luminosity, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Tiled CRA Oklab color correction (WASM export)
 *
 * CRA with overlapping tile-based processing in Oklab color space. Divides the image
 * into blocks with 50% overlap, applies CRA to each block, then blends results using
 * Hamming windows. Combines Oklab's perceptual uniformity with spatial adaptation.
 *
 * Args:
 *     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
 *     input_width, input_height: Input image dimensions
 *     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
 *     ref_width, ref_height: Reference image dimensions
 *     tiled_luminosity: If true, process L channel per-tile before global match
 *     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
 *     dither_mode: 0 = Standard (default), 1 = Serpentine
 *
 * Returns:
 *     Output image as sRGB uint8 (RGBRGB...)
 * @param {Uint8Array} input_data
 * @param {number} input_width
 * @param {number} input_height
 * @param {Uint8Array} ref_data
 * @param {number} ref_width
 * @param {number} ref_height
 * @param {boolean} tiled_luminosity
 * @param {boolean} use_f32_histogram
 * @param {number} dither_mode
 * @returns {Uint8Array}
 */
export function color_correct_tiled_oklab(input_data, input_width, input_height, ref_data, ref_width, ref_height, tiled_luminosity, use_f32_histogram, dither_mode) {
    const ptr0 = passArray8ToWasm0(input_data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(ref_data, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.color_correct_tiled_oklab(ptr0, len0, input_width, input_height, ptr1, len1, ref_width, ref_height, tiled_luminosity, use_f32_histogram, dither_mode);
    var v3 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v3;
}

/**
 * Floyd-Steinberg dithering (WASM export)
 * Matches the existing dither WASM implementation
 * @param {Float32Array} img
 * @param {number} w
 * @param {number} h
 * @returns {Uint8Array}
 */
export function floyd_steinberg_dither_wasm(img, w, h) {
    const ptr0 = passArrayF32ToWasm0(img, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.floyd_steinberg_dither_wasm(ptr0, len0, w, h);
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('cra_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
