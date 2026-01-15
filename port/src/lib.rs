/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// This crate provides WASM-compatible implementations of various color correction
/// algorithms, ported from the original Python scripts.

use wasm_bindgen::prelude::*;
use js_sys;

pub mod basic_lab;
pub mod basic_oklab;
pub mod basic_rgb;
pub mod binary_format;
pub mod color;
mod color_distance;
pub mod colorspace_derived;
pub mod correction;
pub mod cra_lab;
pub mod cra_rgb;
pub mod dither;
pub mod dither_colorspace_aware;
pub mod dither_colorspace_lab;
pub mod dither_colorspace_luminosity;
pub mod dither_common;
mod histogram;
pub mod output;
pub mod pixel;
pub mod rescale;
mod rotation;
pub mod tiled_lab;
mod tiling;

// Re-export dithering function for compatibility with existing WASM code
pub use dither::floyd_steinberg_dither;
use dither_common::DitherMode;
use dither_common::PerceptualSpace;

/// Convert u8 to DitherMode for WASM interface
/// 0 = Floyd-Steinberg Standard
/// 1 = Floyd-Steinberg Serpentine
/// 2 = Jarvis-Judice-Ninke Standard (default for output_dither_mode)
/// 3 = Jarvis-Judice-Ninke Serpentine
/// 4 = Mixed Standard (random kernel, standard scan) (default for histogram_dither_mode)
/// 5 = Mixed Serpentine (random kernel, serpentine scan)
/// 6 = Mixed Random (random kernel and random scan direction)
fn dither_mode_from_u8(mode: u8) -> DitherMode {
    match mode {
        1 => DitherMode::Serpentine,
        2 => DitherMode::JarvisStandard,
        3 => DitherMode::JarvisSerpentine,
        4 => DitherMode::MixedStandard,
        5 => DitherMode::MixedSerpentine,
        6 => DitherMode::MixedRandom,
        _ => DitherMode::Standard,
    }
}

/// Convert u8 to PerceptualSpace for WASM interface
/// 0 = CIELAB with CIE76 (simple Euclidean)
/// 1 = OKLab (default, recommended)
/// 2 = CIELAB with CIE94 (weighted distance)
/// 3 = CIELAB with CIEDE2000 (most accurate)
/// 4 = Linear RGB (NOT RECOMMENDED - for testing only)
/// 5 = Y'CbCr (NOT RECOMMENDED - for testing only)
fn perceptual_space_from_u8(space: u8) -> PerceptualSpace {
    match space {
        0 => PerceptualSpace::LabCIE76,
        2 => PerceptualSpace::LabCIE94,
        3 => PerceptualSpace::LabCIEDE2000,
        4 => PerceptualSpace::LinearRGB,
        5 => PerceptualSpace::YCbCr,
        _ => PerceptualSpace::OkLab,  // 1 or any other value defaults to OkLab
    }
}

/// Floyd-Steinberg dithering (WASM export)
/// Matches the existing dither WASM implementation
#[wasm_bindgen]
pub fn floyd_steinberg_dither_wasm(img: Vec<f32>, w: usize, h: usize) -> Vec<u8> {
    dither::floyd_steinberg_dither(&img, w, h)
}

/// Floyd-Steinberg dithering with configurable bit depth (WASM export)
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     w: image width
///     h: image height
///     bits: output bit depth (1-8), controls number of quantization levels
///
/// Returns:
///     flat array of u8 values. Lower bit depths are represented in uint8
///     by extending bits through repetition (e.g., 3-bit value ABC becomes ABCABCAB).
#[wasm_bindgen]
pub fn floyd_steinberg_dither_bits_wasm(img: Vec<f32>, w: usize, h: usize, bits: u8) -> Vec<u8> {
    dither::floyd_steinberg_dither_bits(&img, w, h, bits)
}

/// Dithering with selectable mode and bit depth (WASM export)
///
/// Args:
///     img: flat array of f32 values in range [0, 255]
///     w: image width
///     h: image height
///     mode: dither mode (0-6, see dither_mode_from_u8)
///     seed: random seed for mixed modes
///     bits: output bit depth (1-8), controls number of quantization levels
///
/// Returns:
///     flat array of u8 values. Lower bit depths are represented in uint8
///     by extending bits through repetition.
#[wasm_bindgen]
pub fn dither_with_mode_wasm(img: Vec<f32>, w: usize, h: usize, mode: u8, seed: u32, bits: u8) -> Vec<u8> {
    dither::dither_with_mode_bits(&img, w, h, dither_mode_from_u8(mode), seed, bits)
}

/// Unified dithering output with technique selection (WASM export)
///
/// Single entry point for all dithering techniques. Takes interleaved sRGB 0-255 input.
///
/// Args:
///     rgb: Interleaved RGB as f32 values in range [0, 255] (RGBRGB...)
///     w: image width
///     h: image height
///     bits_r, bits_g, bits_b: output bit depth per channel (1-8)
///     technique: 0 = None (no dithering), 1 = PerChannel, 2 = ColorspaceAware
///     mode: dither mode (0-6) - used for PerChannel and ColorspaceAware
///     space: perceptual space (0-5) - only used for ColorspaceAware
///     seed: random seed for mixed modes
///
/// Returns:
///     Interleaved RGB uint8 array (RGBRGB...)
#[wasm_bindgen]
pub fn dither_output_wasm(
    rgb: Vec<f32>,
    w: usize,
    h: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    technique: u8,
    mode: u8,
    space: u8,
    seed: u32,
) -> Vec<u8> {
    use dither_common::OutputTechnique;

    let dither_mode = dither_mode_from_u8(mode);
    let perceptual_space = perceptual_space_from_u8(space);

    let output_technique = match technique {
        0 => OutputTechnique::None,
        1 => OutputTechnique::PerChannel { mode: dither_mode },
        _ => OutputTechnique::ColorspaceAware {
            mode: dither_mode,
            space: perceptual_space,
        },
    };

    // Convert interleaved to Pixel4
    let pixels = w * h;
    let mut pixel4_data: Vec<pixel::Pixel4> = Vec::with_capacity(pixels);
    for i in 0..pixels {
        pixel4_data.push([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 0.0]);
    }

    output::dither_output_interleaved(
        &pixel4_data,
        w,
        h,
        bits_r,
        bits_g,
        bits_b,
        output_technique,
        seed,
    )
}

/// Color space aware dithering for grayscale images (WASM export)
///
/// Treats grayscale input as RGB=(L,L,L) for perceptual distance calculation,
/// with error diffusion in linear luminosity space.
/// Input is sRGB gamma-encoded grayscale (0-255).
///
/// Uses Floyd-Steinberg algorithm with Standard (left-to-right) scanning.
/// For other algorithms, use colorspace_aware_dither_gray_with_mode_wasm.
///
/// Args:
///     gray_channel: Grayscale channel as f32 values in range [0, 255]
///     w: image width
///     h: image height
///     bits: output bit depth (1-8)
///     space: perceptual space (0 = CIELAB/CIE76, 1 = OKLab, 2 = CIELAB/CIE94, 3 = CIELAB/CIEDE2000, 4 = LinearRGB, 5 = YCbCr)
///
/// Returns:
///     Grayscale uint8 array
#[wasm_bindgen]
pub fn colorspace_aware_dither_gray_wasm(
    gray_channel: Vec<f32>,
    w: usize,
    h: usize,
    bits: u8,
    space: u8,
) -> Vec<u8> {
    dither_colorspace_luminosity::colorspace_aware_dither_gray(
        &gray_channel,
        w,
        h,
        bits,
        perceptual_space_from_u8(space),
    )
}

/// Color space aware dithering for grayscale with selectable algorithm (WASM export)
///
/// Treats grayscale input as RGB=(L,L,L) for perceptual distance calculation,
/// with error diffusion in linear luminosity space.
/// Input is sRGB gamma-encoded grayscale (0-255).
///
/// Supports the same 7 dithering modes as the RGB version:
/// - 0: Floyd-Steinberg Standard
/// - 1: Floyd-Steinberg Serpentine
/// - 2: Jarvis-Judice-Ninke Standard
/// - 3: Jarvis-Judice-Ninke Serpentine
/// - 4: Mixed Standard
/// - 5: Mixed Serpentine
/// - 6: Mixed Random
///
/// Args:
///     gray_channel: Grayscale channel as f32 values in range [0, 255]
///     w: image width
///     h: image height
///     bits: output bit depth (1-8)
///     space: perceptual space (0 = CIELAB/CIE76, 1 = OKLab, 2 = CIELAB/CIE94, 3 = CIELAB/CIEDE2000, 4 = LinearRGB, 5 = YCbCr)
///     mode: dither mode (0-6, see above)
///     seed: random seed for mixed modes
///
/// Returns:
///     Grayscale uint8 array
#[wasm_bindgen]
pub fn colorspace_aware_dither_gray_with_mode_wasm(
    gray_channel: Vec<f32>,
    w: usize,
    h: usize,
    bits: u8,
    space: u8,
    mode: u8,
    seed: u32,
) -> Vec<u8> {
    use dither_colorspace_luminosity::colorspace_aware_dither_gray_with_mode;
    use dither_colorspace_aware::DitherMode as CSDitherMode;

    let cs_mode = match mode {
        1 => CSDitherMode::Serpentine,
        2 => CSDitherMode::JarvisStandard,
        3 => CSDitherMode::JarvisSerpentine,
        4 => CSDitherMode::MixedStandard,
        5 => CSDitherMode::MixedSerpentine,
        6 => CSDitherMode::MixedRandom,
        _ => CSDitherMode::Standard,
    };

    colorspace_aware_dither_gray_with_mode(
        &gray_channel,
        w,
        h,
        bits,
        perceptual_space_from_u8(space),
        cs_mode,
        seed,
    )
}

// ============================================================================
// Unified Color Correction API (WASM export)
// ============================================================================

/// Convert u8 to ColorCorrectionMethod for WASM interface
/// 0 = BasicLab (keep_luminosity = luminosity_flag)
/// 1 = BasicRgb
/// 2 = BasicOklab (keep_luminosity = luminosity_flag)
/// 3 = CraLab (keep_luminosity = luminosity_flag)
/// 4 = CraRgb (use_perceptual = luminosity_flag)
/// 5 = CraOklab (keep_luminosity = luminosity_flag)
/// 6 = TiledLab (tiled_luminosity = luminosity_flag)
/// 7 = TiledOklab (tiled_luminosity = luminosity_flag)
fn correction_method_from_u8(
    method: u8,
    luminosity_flag: bool,
) -> dither_common::ColorCorrectionMethod {
    use dither_common::ColorCorrectionMethod;
    match method {
        0 => ColorCorrectionMethod::BasicLab {
            keep_luminosity: luminosity_flag,
        },
        1 => ColorCorrectionMethod::BasicRgb,
        2 => ColorCorrectionMethod::BasicOklab {
            keep_luminosity: luminosity_flag,
        },
        3 => ColorCorrectionMethod::CraLab {
            keep_luminosity: luminosity_flag,
        },
        4 => ColorCorrectionMethod::CraRgb {
            use_perceptual: luminosity_flag,
        },
        5 => ColorCorrectionMethod::CraOklab {
            keep_luminosity: luminosity_flag,
        },
        6 => ColorCorrectionMethod::TiledLab {
            tiled_luminosity: luminosity_flag,
        },
        _ => ColorCorrectionMethod::TiledOklab {
            tiled_luminosity: luminosity_flag,
        },
    }
}

/// Convert u8 to HistogramMode for WASM interface
/// 0 = Binned (uint8 256-bin histogram)
/// 1 = EndpointAligned (f32, preserves reference min/max)
/// 2 = MidpointAligned (f32, statistically correct quantiles)
fn histogram_mode_from_u8(mode: u8) -> dither_common::HistogramMode {
    use dither_common::HistogramMode;
    match mode {
        0 => HistogramMode::Binned,
        2 => HistogramMode::MidpointAligned,
        _ => HistogramMode::EndpointAligned,
    }
}

/// Unified color correction (WASM export)
///
/// Single entry point for all color correction methods. Takes linear RGB f32 (0-1)
/// input and returns linear RGB f32 (0-1) output.
///
/// Pipeline: caller handles sRGB u8 → f32 0-255 → f32 0-1 → linear conversion
/// before calling this, and linear → sRGB → f32 0-255 → dither after.
///
/// Args:
///     input_linear: Input image as interleaved linear RGB f32 (0-1 range, RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_linear: Reference image as interleaved linear RGB f32 (0-1 range, RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     method: Color correction method (0-7, see correction_method_from_u8)
///     luminosity_flag: Method-specific flag (keep_luminosity, use_perceptual, or tiled_luminosity)
///     histogram_mode: 0 = binned, 1 = endpoint-aligned, 2 = midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (0-6)
///     colorspace_aware_histogram: Enable colorspace-aware histogram dithering (CRA/Tiled only)
///     histogram_distance_space: Perceptual space for histogram dithering (0-5)
///
/// Returns:
///     Output image as interleaved linear RGB f32 (0-1 range, RGBRGB...)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn color_correct_wasm(
    input_linear: Vec<f32>,
    input_width: usize,
    input_height: usize,
    ref_linear: Vec<f32>,
    ref_width: usize,
    ref_height: usize,
    method: u8,
    luminosity_flag: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    colorspace_aware_histogram: bool,
    histogram_distance_space: u8,
) -> Vec<f32> {
    use correction::HistogramOptions;

    // Convert interleaved f32 to Pixel4
    let input_pixels = input_width * input_height;
    let ref_pixels = ref_width * ref_height;

    let input_pixel4: Vec<pixel::Pixel4> = (0..input_pixels)
        .map(|i| [input_linear[i * 3], input_linear[i * 3 + 1], input_linear[i * 3 + 2], 0.0])
        .collect();

    let ref_pixel4: Vec<pixel::Pixel4> = (0..ref_pixels)
        .map(|i| [ref_linear[i * 3], ref_linear[i * 3 + 1], ref_linear[i * 3 + 2], 0.0])
        .collect();

    // Build method enum
    let correction_method = correction_method_from_u8(method, luminosity_flag);

    // Build histogram options
    let histogram_options = HistogramOptions {
        mode: histogram_mode_from_u8(histogram_mode),
        dither_mode: dither_mode_from_u8(histogram_dither_mode),
        colorspace_aware: colorspace_aware_histogram,
        colorspace_aware_space: perceptual_space_from_u8(histogram_distance_space),
    };

    // Perform color correction
    let result = correction::color_correct(
        &input_pixel4,
        &ref_pixel4,
        input_width,
        input_height,
        ref_width,
        ref_height,
        correction_method,
        histogram_options,
    );

    // Convert Pixel4 back to interleaved f32
    let mut output = vec![0.0f32; input_pixels * 3];
    for (i, pixel) in result.iter().enumerate() {
        output[i * 3] = pixel[0];
        output[i * 3 + 1] = pixel[1];
        output[i * 3 + 2] = pixel[2];
    }

    output
}

// ============================================================================
// Binary Format Encoding (WASM exports)
// ============================================================================

/// Check if a format string is valid (WASM export)
/// Accepts formats like "RGB565", "RGB332", "L4", "L8", etc.
#[wasm_bindgen]
pub fn is_valid_format_wasm(format: &str) -> bool {
    binary_format::is_valid_format(format)
}

/// Check if a format supports binary output (WASM export)
/// Binary output requires 1, 2, 4, 8, 16, 18 (RGB666), 24, or 32 bits per pixel
#[wasm_bindgen]
pub fn format_supports_binary_wasm(format: &str) -> bool {
    binary_format::format_supports_binary(format)
}

/// Check if a format is RGB666 (special 4-pixels-to-9-bytes packing) (WASM export)
#[wasm_bindgen]
pub fn format_is_rgb666_wasm(format: &str) -> bool {
    binary_format::ColorFormat::parse(format)
        .map(|f| f.is_rgb666())
        .unwrap_or(false)
}

/// Get total bits per pixel for a format (WASM export)
/// Returns 0 if format is invalid
#[wasm_bindgen]
pub fn format_total_bits_wasm(format: &str) -> u8 {
    binary_format::format_total_bits(format).unwrap_or(0)
}

/// Check if format is grayscale (WASM export)
#[wasm_bindgen]
pub fn format_is_grayscale_wasm(format: &str) -> bool {
    binary_format::format_is_grayscale(format)
}

/// Encode RGB data to packed binary format (WASM export)
///
/// Takes separate R, G, B channels (uint8 arrays) and encodes to packed binary.
/// Output is little-endian for multi-byte formats.
///
/// Args:
///     r_data, g_data, b_data: Separate channel data as uint8 arrays
///     width, height: Image dimensions
///     bits_r, bits_g, bits_b: Bits per channel
///
/// fill: 0 = Black (zeros), 1 = Repeat last pixel (only affects RGB666 partial groups)
///
/// Returns:
///     Packed binary data as uint8 array
#[wasm_bindgen]
pub fn encode_rgb_packed_wasm(
    r_data: Vec<u8>,
    g_data: Vec<u8>,
    b_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    fill: u8,
) -> Vec<u8> {
    let fill_mode = if fill == 0 {
        binary_format::StrideFill::Black
    } else {
        binary_format::StrideFill::Repeat
    };
    binary_format::encode_rgb_packed(&r_data, &g_data, &b_data, width, height, bits_r, bits_g, bits_b, fill_mode)
}

/// Encode RGB data to row-aligned binary format (WASM export)
///
/// Each row is padded to a byte boundary.
/// Output is little-endian for multi-byte formats.
#[wasm_bindgen]
pub fn encode_rgb_row_aligned_wasm(
    r_data: Vec<u8>,
    g_data: Vec<u8>,
    b_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
) -> Vec<u8> {
    binary_format::encode_rgb_row_aligned(&r_data, &g_data, &b_data, width, height, bits_r, bits_g, bits_b)
}

/// Encode grayscale data to packed binary format (WASM export)
///
/// Args:
///     gray_data: Grayscale data as uint8 array
///     width, height: Image dimensions
///     bits: Bits per pixel (1-8)
///
/// Returns:
///     Packed binary data as uint8 array
#[wasm_bindgen]
pub fn encode_gray_packed_wasm(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_gray_packed(&gray_data, width, height, bits)
}

/// Encode grayscale data to row-aligned binary format (WASM export)
///
/// Each row is padded to a byte boundary.
#[wasm_bindgen]
pub fn encode_gray_row_aligned_wasm(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_gray_row_aligned(&gray_data, width, height, bits)
}

/// Encode a single channel to packed binary format (WASM export)
///
/// Useful for per-channel binary output (e.g., R channel only at 5 bits)
#[wasm_bindgen]
pub fn encode_channel_packed_wasm(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_channel_packed(&channel_data, width, height, bits)
}

/// Encode a single channel to row-aligned binary format (WASM export)
#[wasm_bindgen]
pub fn encode_channel_row_aligned_wasm(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
) -> Vec<u8> {
    binary_format::encode_channel_row_aligned(&channel_data, width, height, bits)
}

// ============================================================================
// Stride-aligned Binary Format Encoding (WASM exports)
// ============================================================================

/// Validate stride alignment value (WASM export)
/// Valid values are powers of 2 from 1 to 128
#[wasm_bindgen]
pub fn is_valid_stride_wasm(stride: usize) -> bool {
    binary_format::is_valid_stride(stride)
}

/// Encode RGB data to row-aligned binary format with configurable stride (WASM export)
///
/// stride: Row stride alignment in bytes (must be power of 2, 1-128)
/// fill: How to fill padding (0 = black/zeros, 1 = repeat last pixel)
#[wasm_bindgen]
pub fn encode_rgb_row_aligned_stride_wasm(
    r_data: Vec<u8>,
    g_data: Vec<u8>,
    b_data: Vec<u8>,
    width: usize,
    height: usize,
    bits_r: u8,
    bits_g: u8,
    bits_b: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_rgb_row_aligned_stride(&r_data, &g_data, &b_data, width, height, bits_r, bits_g, bits_b, stride, binary_format::StrideFill::from_u8(fill))
}

/// Encode grayscale data to row-aligned binary format with configurable stride (WASM export)
///
/// stride: Row stride alignment in bytes (must be power of 2, 1-128)
/// fill: How to fill padding (0 = black/zeros, 1 = repeat last pixel)
#[wasm_bindgen]
pub fn encode_gray_row_aligned_stride_wasm(
    gray_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_gray_row_aligned_stride(&gray_data, width, height, bits, stride, binary_format::StrideFill::from_u8(fill))
}

/// Encode a single channel to row-aligned binary format with configurable stride (WASM export)
///
/// stride: Row stride alignment in bytes (must be power of 2, 1-128)
/// fill: How to fill padding (0 = black/zeros, 1 = repeat last pixel)
#[wasm_bindgen]
pub fn encode_channel_row_aligned_stride_wasm(
    channel_data: Vec<u8>,
    width: usize,
    height: usize,
    bits: u8,
    stride: usize,
    fill: u8,
) -> Vec<u8> {
    binary_format::encode_channel_row_aligned_stride(&channel_data, width, height, bits, stride, binary_format::StrideFill::from_u8(fill))
}

// ============================================================================
// Color Space Conversion WASM Exports (Atomic API)
// ============================================================================
//
// Pipeline: u8 → unpack → sRGB f32 → linear → [process] → sRGB f32 → dither/pack
//
// Functions:
// - unpack_u8_to_f32_wasm: u8 0-255 → f32 0-255 (type conversion only)
// - normalize_f32_wasm: f32 0-255 → f32 0-1 (normalization only)
// - denormalize_f32_wasm: f32 0-1 → f32 0-255 (denormalization only)
// - srgb_to_linear_f32_wasm: f32 0-1 sRGB → f32 0-1 linear (gamma only)
// - linear_to_srgb_f32_wasm: f32 0-1 linear → f32 0-1 sRGB (gamma only)
// - linear_rgb_to_grayscale_wasm: linear RGB → linear L
// - histogram_match_channel_wasm: single channel histogram matching
// - histogram_match_rgb_wasm: RGB histogram matching (3 channels)
//
// The only time linear conversion is skipped is when there's no processing
// (dither only).

/// Unpack u8 image data to f32 (no normalization, no color space conversion)
/// Input: interleaved RGB u8 (0-255)
/// Output: interleaved RGB f32 (0-255 scale, still sRGB)
#[wasm_bindgen]
pub fn unpack_u8_to_f32_wasm(data: Vec<u8>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut result = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        result[i * 3] = data[i * 3] as f32;
        result[i * 3 + 1] = data[i * 3 + 1] as f32;
        result[i * 3 + 2] = data[i * 3 + 2] as f32;
    }

    result
}

/// Normalize f32 from 0-255 to 0-1 range (no color space conversion)
/// Input: interleaved RGB f32 (0-255 scale)
/// Output: interleaved RGB f32 (0-1 scale)
#[wasm_bindgen]
pub fn normalize_f32_wasm(data: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut result = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        result[i * 3] = data[i * 3] / 255.0;
        result[i * 3 + 1] = data[i * 3 + 1] / 255.0;
        result[i * 3 + 2] = data[i * 3 + 2] / 255.0;
    }

    result
}

/// Denormalize f32 from 0-1 to 0-255 range (no color space conversion)
/// Input: interleaved RGB f32 (0-1 scale)
/// Output: interleaved RGB f32 (0-255 scale)
#[wasm_bindgen]
pub fn denormalize_f32_wasm(data: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut result = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        result[i * 3] = data[i * 3] * 255.0;
        result[i * 3 + 1] = data[i * 3 + 1] * 255.0;
        result[i * 3 + 2] = data[i * 3 + 2] * 255.0;
    }

    result
}

/// Convert sRGB f32 (0-1) to linear RGB f32 (0-1)
/// Input: interleaved RGB f32 (0-1 scale, sRGB)
/// Output: interleaved RGB f32 (0-1 scale, linear)
#[wasm_bindgen]
pub fn srgb_to_linear_f32_wasm(srgb: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut linear = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        linear[i * 3] = color::srgb_to_linear_single(srgb[i * 3]);
        linear[i * 3 + 1] = color::srgb_to_linear_single(srgb[i * 3 + 1]);
        linear[i * 3 + 2] = color::srgb_to_linear_single(srgb[i * 3 + 2]);
    }

    linear
}

/// Convert linear RGB f32 (0-1) to sRGB f32 (0-1)
/// Input: interleaved RGB f32 (0-1 scale, linear)
/// Output: interleaved RGB f32 (0-1 scale, sRGB)
#[wasm_bindgen]
pub fn linear_to_srgb_f32_wasm(linear: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut srgb = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        srgb[i * 3] = color::linear_to_srgb_single(linear[i * 3]);
        srgb[i * 3 + 1] = color::linear_to_srgb_single(linear[i * 3 + 1]);
        srgb[i * 3 + 2] = color::linear_to_srgb_single(linear[i * 3 + 2]);
    }

    srgb
}

/// Convert linear RGB to linear grayscale (luminance)
/// Input: interleaved RGB f32 (0-1 scale, linear)
/// Output: single channel f32 (0-1 scale, linear luminance)
/// Uses Rec.709/BT.709 luminance coefficients
#[wasm_bindgen]
pub fn linear_rgb_to_grayscale_wasm(linear_rgb: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut gray = Vec::with_capacity(pixels);

    for i in 0..pixels {
        let r = linear_rgb[i * 3];
        let g = linear_rgb[i * 3 + 1];
        let b = linear_rgb[i * 3 + 2];
        gray.push(color::linear_rgb_to_luminance(r, g, b));
    }

    gray
}

/// Convert linear grayscale (0-1) to sRGB grayscale (0-1)
/// Input: single channel f32 (0-1 scale, linear)
/// Output: single channel f32 (0-1 scale, sRGB)
#[wasm_bindgen]
pub fn linear_gray_to_srgb_f32_wasm(linear_gray: Vec<f32>) -> Vec<f32> {
    linear_gray
        .iter()
        .map(|&l| color::linear_to_srgb_single(l))
        .collect()
}

/// Denormalize grayscale f32 from 0-1 to 0-255 range
/// Input: single channel f32 (0-1 scale)
/// Output: single channel f32 (0-255 scale)
#[wasm_bindgen]
pub fn denormalize_gray_f32_wasm(gray: Vec<f32>) -> Vec<f32> {
    gray.iter().map(|&g| g * 255.0).collect()
}

// ============================================================================
// Histogram Matching WASM Exports (Atomic API)
// ============================================================================

/// Match histogram of a single channel
/// Input: source and reference channels as f32 (any range, typically 0-1 for linear)
/// Output: matched source with histogram matching reference distribution
/// Uses linear interpolation and midpoint alignment by default
#[wasm_bindgen]
pub fn histogram_match_channel_wasm(source: Vec<f32>, reference: Vec<f32>, seed: u32) -> Vec<f32> {
    histogram::match_histogram_f32(
        &source,
        &reference,
        histogram::InterpolationMode::Linear,
        histogram::AlignmentMode::Midpoint,
        seed,
    )
}

/// Match histogram of RGB image (3 channels independently)
/// Input: interleaved RGB f32 (linear, 0-1 scale)
/// Output: interleaved RGB f32 (linear, 0-1 scale)
#[wasm_bindgen]
pub fn histogram_match_rgb_wasm(
    source: Vec<f32>,
    reference: Vec<f32>,
    src_width: usize,
    src_height: usize,
    ref_width: usize,
    ref_height: usize,
    seed: u32,
) -> Vec<f32> {
    let src_pixels = src_width * src_height;
    let ref_pixels = ref_width * ref_height;

    // Extract channels
    let mut src_r = Vec::with_capacity(src_pixels);
    let mut src_g = Vec::with_capacity(src_pixels);
    let mut src_b = Vec::with_capacity(src_pixels);
    let mut ref_r = Vec::with_capacity(ref_pixels);
    let mut ref_g = Vec::with_capacity(ref_pixels);
    let mut ref_b = Vec::with_capacity(ref_pixels);

    for i in 0..src_pixels {
        src_r.push(source[i * 3]);
        src_g.push(source[i * 3 + 1]);
        src_b.push(source[i * 3 + 2]);
    }

    for i in 0..ref_pixels {
        ref_r.push(reference[i * 3]);
        ref_g.push(reference[i * 3 + 1]);
        ref_b.push(reference[i * 3 + 2]);
    }

    // Match each channel (use different seeds to reduce correlation)
    let matched_r = histogram::match_histogram_f32(
        &src_r,
        &ref_r,
        histogram::InterpolationMode::Linear,
        histogram::AlignmentMode::Midpoint,
        seed,
    );
    let matched_g = histogram::match_histogram_f32(
        &src_g,
        &ref_g,
        histogram::InterpolationMode::Linear,
        histogram::AlignmentMode::Midpoint,
        seed.wrapping_add(1),
    );
    let matched_b = histogram::match_histogram_f32(
        &src_b,
        &ref_b,
        histogram::InterpolationMode::Linear,
        histogram::AlignmentMode::Midpoint,
        seed.wrapping_add(2),
    );

    // Interleave result
    let mut result = vec![0.0f32; src_pixels * 3];
    for i in 0..src_pixels {
        result[i * 3] = matched_r[i];
        result[i * 3 + 1] = matched_g[i];
        result[i * 3 + 2] = matched_b[i];
    }

    result
}

// ============================================================================
// Image Rescaling WASM Exports
// ============================================================================

/// Rescale method enum for WASM
/// 0 = Bilinear
/// 1 = Lanczos3
fn rescale_method_from_u8(method: u8) -> rescale::RescaleMethod {
    match method {
        1 => rescale::RescaleMethod::Lanczos3,
        _ => rescale::RescaleMethod::Bilinear,
    }
}

fn scale_mode_from_u8(mode: u8) -> rescale::ScaleMode {
    match mode {
        1 => rescale::ScaleMode::UniformWidth,
        2 => rescale::ScaleMode::UniformHeight,
        _ => rescale::ScaleMode::Independent,
    }
}

/// Rescale linear RGB image (interleaved, 0-1 range)
/// Returns interleaved RGB as f32 in 0-1 range
/// This is the atomic rescale function - use with linear data.
/// scale_mode: 0=independent, 1=uniform from width, 2=uniform from height
#[wasm_bindgen]
pub fn rescale_linear_rgb_wasm(
    linear: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
) -> Vec<f32> {
    let method = rescale_method_from_u8(method);
    let scale_mode = scale_mode_from_u8(scale_mode);
    rescale::rescale_rgb_interleaved(&linear, src_width, src_height, dst_width, dst_height, method, scale_mode)
}

/// Rescale linear grayscale image (single channel, 0-1 range)
/// Returns single channel f32 in 0-1 range
/// scale_mode: 0=independent, 1=uniform from width, 2=uniform from height
#[wasm_bindgen]
pub fn rescale_linear_gray_wasm(
    linear_gray: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
) -> Vec<f32> {
    let method = rescale_method_from_u8(method);
    let scale_mode = scale_mode_from_u8(scale_mode);
    rescale::rescale_channel_uniform(&linear_gray, src_width, src_height, dst_width, dst_height, method, scale_mode)
}

/// Calculate target dimensions preserving aspect ratio
/// Returns [width, height]
/// If both target_width and target_height are 0, returns source dimensions
#[wasm_bindgen]
pub fn calculate_dimensions_wasm(
    src_width: usize,
    src_height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<usize> {
    let tw = if target_width == 0 { None } else { Some(target_width) };
    let th = if target_height == 0 { None } else { Some(target_height) };
    let (w, h) = rescale::calculate_target_dimensions(src_width, src_height, tw, th);
    vec![w, h]
}

/// Rescale linear RGB image with progress callback (interleaved, 0-1 range)
/// Returns interleaved RGB as f32 in 0-1 range
/// The progress callback receives a f32 value from 0.0 (before start) to 1.0 (after completion)
/// Progress is reported row-by-row.
/// scale_mode: 0=independent, 1=uniform from width, 2=uniform from height
#[wasm_bindgen]
pub fn rescale_linear_rgb_with_progress_wasm(
    linear: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
    scale_mode: u8,
    progress_callback: &js_sys::Function,
) -> Vec<f32> {
    let method = rescale_method_from_u8(method);
    let scale_mode = scale_mode_from_u8(scale_mode);

    // Wrap the JS callback
    let js_this = wasm_bindgen::JsValue::NULL;
    let callback = progress_callback.clone();

    let mut progress_fn = |progress: f32| {
        let _ = callback.call1(&js_this, &wasm_bindgen::JsValue::from_f64(progress as f64));
    };

    rescale::rescale_rgb_interleaved_with_progress(
        &linear, src_width, src_height, dst_width, dst_height, method, scale_mode,
        Some(&mut progress_fn)
    )
}

// ============================================================================
// Image Decoding WASM Exports (Precise 16-bit and ICC support)
// ============================================================================
//
// These functions allow decoding images directly in WASM, bypassing the
// browser's Canvas API which loses precision (converts to 8-bit) and ICC info.

pub mod decode;

/// Decode image from raw file bytes
/// Returns: [width, height, has_icc (0/1), is_16bit (0/1), ...pixel_data]
/// Pixel data is interleaved RGB f32 in 0-1 range (normalized, NOT linearized)
/// The caller should apply srgb_to_linear if needed.
#[wasm_bindgen]
pub fn decode_image_wasm(file_bytes: Vec<u8>) -> Result<Vec<f32>, JsValue> {
    decode::decode_image_to_f32(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))
}

/// Decode image directly to sRGB f32 0-255 scale (no 0-1 intermediate)
/// Use this for sRGB-direct paths where no color processing is needed.
/// Returns: [width, height, has_icc (0/1), is_16bit (0/1), ...pixel_data]
/// Pixel data is interleaved RGB f32 in 0-255 range, ready for dithering.
#[wasm_bindgen]
pub fn decode_image_srgb_255_wasm(file_bytes: Vec<u8>) -> Result<Vec<f32>, JsValue> {
    decode::decode_image_to_srgb_255(&file_bytes)
        .map_err(|e| JsValue::from_str(&e))
}

/// Get ICC profile from raw file bytes (if present)
/// Returns empty Vec if no profile found
#[wasm_bindgen]
pub fn extract_icc_profile_wasm(file_bytes: Vec<u8>) -> Vec<u8> {
    decode::extract_icc_profile(&file_bytes).unwrap_or_default()
}

/// Check if ICC profile is effectively sRGB
/// Returns true if profile is sRGB or sRGB-compatible
#[wasm_bindgen]
pub fn is_icc_profile_srgb_wasm(icc_bytes: Vec<u8>) -> bool {
    decode::is_profile_srgb(&icc_bytes)
}

/// Transform image from ICC profile to linear sRGB
/// Input: interleaved RGB f32 (0-1 range, in source color space)
/// Output: interleaved RGB f32 (0-1 range, linear sRGB)
#[wasm_bindgen]
pub fn transform_icc_to_linear_srgb_wasm(
    pixels: Vec<f32>,
    width: usize,
    height: usize,
    icc_bytes: Vec<u8>,
) -> Result<Vec<f32>, JsValue> {
    decode::transform_icc_to_linear_srgb(&pixels, width, height, &icc_bytes)
        .map_err(|e| JsValue::from_str(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to convert u8 to linear f32 for testing
    fn u8_to_linear_f32(data: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut pixels = pixel::srgb_u8_to_pixels(data);
        color::normalize_inplace(&mut pixels);
        color::srgb_to_linear_inplace(&mut pixels);
        // Convert to interleaved f32
        pixels.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
    }

    // Helper to convert linear f32 to sRGB u8 with dithering
    fn linear_f32_to_u8(data: &[f32], width: usize, height: usize, dither_mode: u8, colorspace_aware: bool, space: u8) -> Vec<u8> {
        // Convert interleaved f32 to Pixel4
        let pixels: usize = width * height;
        let mut pixel4: Vec<pixel::Pixel4> = (0..pixels)
            .map(|i| [data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.0])
            .collect();
        // Finalize to sRGB u8 output
        output::finalize_to_srgb_u8_with_options(
            &mut pixel4,
            width,
            height,
            Some(dither_mode_from_u8(dither_mode)),
            colorspace_aware,
            perceptual_space_from_u8(space),
            0, // seed
        )
    }

    // Helper to call unified color_correct_wasm with full pipeline
    // method: 0=BasicLab, 1=BasicRgb, 2=BasicOklab, 3=CraLab, 4=CraRgb, 5=CraOklab, 6=TiledLab, 7=TiledOklab
    fn cc(
        input: &[u8], iw: usize, ih: usize,
        reference: &[u8], rw: usize, rh: usize,
        method: u8, lum_flag: bool,
        hist_mode: u8, hist_dither: u8,
        ca_hist: bool, hist_space: u8,
        out_dither: u8, ca_out: bool, out_space: u8,
    ) -> Vec<u8> {
        // Step 1: Convert inputs to linear f32
        let input_linear = u8_to_linear_f32(input, iw, ih);
        let ref_linear = u8_to_linear_f32(reference, rw, rh);

        // Step 2: Call color_correct_wasm (now takes linear f32)
        let result_linear = color_correct_wasm(
            input_linear, iw, ih,
            ref_linear, rw, rh,
            method, lum_flag, hist_mode, hist_dither,
            ca_hist, hist_space,
        );

        // Step 3: Convert result back to sRGB u8 with dithering
        linear_f32_to_u8(&result_linear, iw, ih, out_dither, ca_out, out_space)
    }

    #[test]
    fn test_basic_lab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=0 (BasicLab), keep_luminosity=false, histogram_mode=0, histogram_dither=4, output_dither=2
        let result = cc(&input, 2, 2, &reference, 2, 2, 0, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 0, false, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 0, false, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_basic_rgb_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=1 (BasicRgb)
        let result = cc(&input, 2, 2, &reference, 2, 2, 1, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 1, false, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 1, false, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_cra_lab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=3 (CraLab), no color-aware
        let result = cc(&input, 2, 2, &reference, 2, 2, 3, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        // Test with color-aware histogram (hist_space=3 for CIEDE2000)
        let result_ca = cc(&input, 2, 2, &reference, 2, 2, 3, false, 0, 4, true, 3, 2, false, 0);
        assert_eq!(result_ca.len(), 12);

        // Test with color-aware output (out_space=1 for OkLab)
        let result_co = cc(&input, 2, 2, &reference, 2, 2, 3, false, 0, 4, false, 0, 2, true, 1);
        assert_eq!(result_co.len(), 12);

        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 3, false, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 3, false, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_cra_rgb_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=4 (CraRgb), use_perceptual=false
        let result = cc(&input, 2, 2, &reference, 2, 2, 4, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 4, false, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 4, false, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_basic_oklab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=2 (BasicOklab)
        let result = cc(&input, 2, 2, &reference, 2, 2, 2, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 2, false, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 2, false, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_cra_oklab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=5 (CraOklab)
        let result = cc(&input, 2, 2, &reference, 2, 2, 5, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        // Test with color-aware output
        let result_co = cc(&input, 2, 2, &reference, 2, 2, 5, false, 0, 4, false, 0, 2, true, 1);
        assert_eq!(result_co.len(), 12);

        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 5, false, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 5, false, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_tiled_oklab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // method=7 (TiledOklab), tiled_luminosity=true
        let result = cc(&input, 2, 2, &reference, 2, 2, 7, true, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result.len(), 12);

        // Test without tiled luminosity (AB only)
        let result_ab = cc(&input, 2, 2, &reference, 2, 2, 7, false, 0, 4, false, 0, 2, false, 0);
        assert_eq!(result_ab.len(), 12);

        // Test with color-aware output
        let result_co = cc(&input, 2, 2, &reference, 2, 2, 7, true, 0, 4, false, 0, 2, true, 1);
        assert_eq!(result_co.len(), 12);

        let result_f32 = cc(&input, 2, 2, &reference, 2, 2, 7, true, 1, 4, false, 0, 2, false, 0);
        assert_eq!(result_f32.len(), 12);

        let result_mid = cc(&input, 2, 2, &reference, 2, 2, 7, true, 2, 4, false, 0, 2, false, 0);
        assert_eq!(result_mid.len(), 12);
    }
}
