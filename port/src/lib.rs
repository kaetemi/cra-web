/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// This crate provides WASM-compatible implementations of various color correction
/// algorithms, ported from the original Python scripts.

use wasm_bindgen::prelude::*;

pub mod basic_lab;
pub mod basic_oklab;
pub mod basic_rgb;
pub mod binary_format;
pub mod color;
mod color_distance;
pub mod colorspace_derived;
pub mod colorspace_primary;
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
///     technique: 0 = None (no dithering), 1 = PerChannel, 2 = ColorAware
///     mode: dither mode (0-6) - used for PerChannel and ColorAware
///     space: perceptual space (0-5) - only used for ColorAware
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
        _ => OutputTechnique::ColorAware {
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
/// Single entry point for all color correction methods. Takes sRGB uint8 input
/// and returns sRGB uint8 output.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     method: Color correction method (0-7, see correction_method_from_u8)
///     luminosity_flag: Method-specific flag (keep_luminosity, use_perceptual, or tiled_luminosity)
///     histogram_mode: 0 = binned, 1 = endpoint-aligned, 2 = midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (0-6)
///     color_aware_histogram: Enable color-aware histogram dithering (CRA/Tiled only)
///     histogram_distance_space: Perceptual space for histogram dithering (0-5)
///     output_dither_mode: Dither mode for final RGB output (0-6)
///     color_aware_output: Enable color-aware output dithering
///     output_distance_space: Perceptual space for output dithering (0-5)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn color_correct_wasm(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    method: u8,
    luminosity_flag: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    color_aware_histogram: bool,
    histogram_distance_space: u8,
    output_dither_mode: u8,
    color_aware_output: bool,
    output_distance_space: u8,
) -> Vec<u8> {
    use correction::HistogramOptions;

    // Convert sRGB u8 to Pixel4 linear RGB
    let mut input_pixels = pixel::srgb_u8_to_pixels(input_data);
    color::srgb_255_to_linear_inplace(&mut input_pixels);
    let mut ref_pixels = pixel::srgb_u8_to_pixels(ref_data);
    color::srgb_255_to_linear_inplace(&mut ref_pixels);

    // Build method enum
    let correction_method = correction_method_from_u8(method, luminosity_flag);

    // Build histogram options
    let histogram_options = HistogramOptions {
        mode: histogram_mode_from_u8(histogram_mode),
        dither_mode: dither_mode_from_u8(histogram_dither_mode),
        color_aware: color_aware_histogram,
        color_aware_space: perceptual_space_from_u8(histogram_distance_space),
    };

    // Perform color correction
    let mut result = correction::color_correct(
        &input_pixels,
        &ref_pixels,
        input_width,
        input_height,
        ref_width,
        ref_height,
        correction_method,
        histogram_options,
    );

    // Finalize to sRGB u8 output
    output::finalize_to_srgb_u8_with_options(
        &mut result,
        input_width,
        input_height,
        Some(dither_mode_from_u8(output_dither_mode)),
        color_aware_output,
        perceptual_space_from_u8(output_distance_space),
        0, // seed
    )
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
// Color Space Conversion WASM Exports
// ============================================================================

/// Convert sRGB image (interleaved RGB, 0-255) to linear RGB channels (0-1 range)
/// Returns interleaved RGB as f32
#[wasm_bindgen]
pub fn srgb_to_linear_wasm(srgb: Vec<u8>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut linear = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        linear[i * 3] = color::srgb_to_linear_single(srgb[i * 3] as f32 / 255.0);
        linear[i * 3 + 1] = color::srgb_to_linear_single(srgb[i * 3 + 1] as f32 / 255.0);
        linear[i * 3 + 2] = color::srgb_to_linear_single(srgb[i * 3 + 2] as f32 / 255.0);
    }

    linear
}

/// Convert linear RGB (interleaved, 0-1 range) to sRGB (0-255 range)
/// Returns interleaved RGB as f32 in 0-255 range (for dithering)
#[wasm_bindgen]
pub fn linear_to_srgb_wasm(linear: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut srgb = vec![0.0f32; pixels * 3];

    for i in 0..pixels {
        srgb[i * 3] = color::linear_to_srgb_single(linear[i * 3]) * 255.0;
        srgb[i * 3 + 1] = color::linear_to_srgb_single(linear[i * 3 + 1]) * 255.0;
        srgb[i * 3 + 2] = color::linear_to_srgb_single(linear[i * 3 + 2]) * 255.0;
    }

    srgb
}

/// Convert sRGB RGB image to grayscale using proper linear-space luminance computation
/// Input: sRGB interleaved RGB (0-255 as u8)
/// Output: sRGB grayscale (0-255 range as f32, ready for dithering)
///
/// Pipeline: sRGB -> linear RGB -> luminance (Rec.709) -> sRGB
#[wasm_bindgen]
pub fn srgb_to_grayscale_wasm(srgb: Vec<u8>, width: usize, height: usize) -> Vec<f32> {
    color::srgb_interleaved_to_grayscale(&srgb, width * height)
}

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

/// Rescale sRGB image (interleaved RGB, 0-255 as u8)
/// Converts to linear, rescales, converts back to sRGB
/// Returns interleaved RGB as f32 in 0-255 range (ready for dithering)
///
/// Args:
///     srgb: interleaved RGB u8 values
///     src_width, src_height: source dimensions
///     dst_width, dst_height: target dimensions
///     method: 0=Bilinear, 1=Lanczos3
#[wasm_bindgen]
pub fn rescale_srgb_wasm(
    srgb: Vec<u8>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
) -> Vec<f32> {
    let src_pixels = src_width * src_height;
    let method = rescale_method_from_u8(method);

    // Convert sRGB to linear RGB (interleaved f32, 0-1 range)
    let mut linear = vec![0.0f32; src_pixels * 3];
    for i in 0..src_pixels {
        linear[i * 3] = color::srgb_to_linear_single(srgb[i * 3] as f32 / 255.0);
        linear[i * 3 + 1] = color::srgb_to_linear_single(srgb[i * 3 + 1] as f32 / 255.0);
        linear[i * 3 + 2] = color::srgb_to_linear_single(srgb[i * 3 + 2] as f32 / 255.0);
    }

    // Rescale in linear space
    let linear_rescaled = rescale::rescale_rgb_interleaved(
        &linear, src_width, src_height, dst_width, dst_height, method
    );

    // Convert back to sRGB (0-255 range)
    let dst_pixels = dst_width * dst_height;
    let mut result = vec![0.0f32; dst_pixels * 3];
    for i in 0..dst_pixels {
        result[i * 3] = color::linear_to_srgb_single(linear_rescaled[i * 3]) * 255.0;
        result[i * 3 + 1] = color::linear_to_srgb_single(linear_rescaled[i * 3 + 1]) * 255.0;
        result[i * 3 + 2] = color::linear_to_srgb_single(linear_rescaled[i * 3 + 2]) * 255.0;
    }

    result
}

/// Rescale linear RGB image (interleaved, 0-1 range)
/// Returns interleaved RGB as f32 in 0-1 range
#[wasm_bindgen]
pub fn rescale_linear_wasm(
    linear: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
) -> Vec<f32> {
    let method = rescale_method_from_u8(method);
    rescale::rescale_rgb_interleaved(&linear, src_width, src_height, dst_width, dst_height, method)
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

// ============================================================================
// SIMD-friendly WASM Exports (4-channel format)
// ============================================================================

/// Rescale sRGB image using 4-channel SIMD-friendly format
/// Input: flat array of RGBX f32 values (0-255 scale), 4 values per pixel
/// Output: flat array of RGBX f32 values (0-255 scale)
/// Performs: sRGB -> linear -> rescale -> sRGB
#[wasm_bindgen]
pub fn rescale_srgb_4ch_wasm(
    srgb: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
) -> Vec<f32> {
    let src_pixels = src_width * src_height;
    let method = rescale_method_from_u8(method);

    // Convert flat array to Pixel4 array and sRGB to linear
    let mut linear_pixels: Vec<pixel::Pixel4> = Vec::with_capacity(src_pixels);
    for i in 0..src_pixels {
        linear_pixels.push([
            color::srgb_to_linear_single(srgb[i * 4] / 255.0),
            color::srgb_to_linear_single(srgb[i * 4 + 1] / 255.0),
            color::srgb_to_linear_single(srgb[i * 4 + 2] / 255.0),
            srgb[i * 4 + 3], // preserve 4th channel
        ]);
    }

    // Rescale in linear space
    let rescaled = rescale::rescale(
        &linear_pixels, src_width, src_height, dst_width, dst_height, method
    );

    // Convert back to sRGB and flatten
    let dst_pixels = dst_width * dst_height;
    let mut result = Vec::with_capacity(dst_pixels * 4);
    for p in &rescaled {
        result.push(color::linear_to_srgb_single(p[0]) * 255.0);
        result.push(color::linear_to_srgb_single(p[1]) * 255.0);
        result.push(color::linear_to_srgb_single(p[2]) * 255.0);
        result.push(p[3]); // preserve 4th channel
    }

    result
}

/// Rescale linear RGB image using 4-channel SIMD-friendly format
/// Input: flat array of RGBX f32 values (0-1 scale), 4 values per pixel
/// Output: flat array of RGBX f32 values (0-1 scale)
#[wasm_bindgen]
pub fn rescale_linear_4ch_wasm(
    linear: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
) -> Vec<f32> {
    let method = rescale_method_from_u8(method);

    // Convert flat array to Pixel4 array
    let linear_pixels = pixel::interleaved_rgba_to_pixels(&linear);

    // Rescale
    let rescaled = rescale::rescale(
        &linear_pixels, src_width, src_height, dst_width, dst_height, method
    );

    // Flatten back to Vec<f32>
    pixel::pixels_to_interleaved_rgba(&rescaled)
}

/// Rescale single channel (grayscale)
/// Input: flat array of f32 values
/// Output: flat array of f32 values
#[wasm_bindgen]
pub fn rescale_channel_wasm(
    src: Vec<f32>,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    method: u8,
) -> Vec<f32> {
    let method = rescale_method_from_u8(method);
    rescale::rescale_channel(&src, src_width, src_height, dst_width, dst_height, method)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to call unified color_correct_wasm
    // method: 0=BasicLab, 1=BasicRgb, 2=BasicOklab, 3=CraLab, 4=CraRgb, 5=CraOklab, 6=TiledLab, 7=TiledOklab
    fn cc(
        input: &[u8], iw: usize, ih: usize,
        reference: &[u8], rw: usize, rh: usize,
        method: u8, lum_flag: bool,
        hist_mode: u8, hist_dither: u8,
        ca_hist: bool, hist_space: u8,
        out_dither: u8, ca_out: bool, out_space: u8,
    ) -> Vec<u8> {
        color_correct_wasm(
            input, iw, ih, reference, rw, rh,
            method, lum_flag, hist_mode, hist_dither,
            ca_hist, hist_space, out_dither, ca_out, out_space,
        )
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
