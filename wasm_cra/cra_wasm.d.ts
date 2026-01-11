/* tslint:disable */
/* eslint-disable */

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
 */
export function color_correct_basic_lab(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, keep_luminosity: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_basic_oklab(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, keep_luminosity: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_basic_rgb(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_cra_lab(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, keep_luminosity: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_cra_oklab(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, keep_luminosity: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_cra_rgb(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, use_perceptual: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_tiled_lab(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, tiled_luminosity: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

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
 */
export function color_correct_tiled_oklab(input_data: Uint8Array, input_width: number, input_height: number, ref_data: Uint8Array, ref_width: number, ref_height: number, tiled_luminosity: boolean, use_f32_histogram: boolean, dither_mode: number): Uint8Array;

/**
 * Floyd-Steinberg dithering (WASM export)
 * Matches the existing dither WASM implementation
 */
export function floyd_steinberg_dither_wasm(img: Float32Array, w: number, h: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly color_correct_basic_lab: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly color_correct_basic_oklab: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly color_correct_basic_rgb: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
  readonly color_correct_cra_lab: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly color_correct_cra_oklab: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly color_correct_cra_rgb: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly color_correct_tiled_lab: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly color_correct_tiled_oklab: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
  readonly floyd_steinberg_dither_wasm: (a: number, b: number, c: number, d: number) => [number, number];
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
