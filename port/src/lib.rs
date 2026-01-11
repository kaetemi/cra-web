/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// This crate provides WASM-compatible implementations of various color correction
/// algorithms, ported from the original Python scripts.

use wasm_bindgen::prelude::*;

mod basic_lab;
mod basic_oklab;
mod basic_rgb;
mod color;
mod cra_lab;
mod cra_oklab;
mod cra_rgb;
mod dither;
mod histogram;
mod rotation;
mod tiled_lab;
mod tiled_oklab;
mod tiling;

// Re-export dithering function for compatibility with existing WASM code
pub use dither::floyd_steinberg_dither;
use dither::DitherMode;

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

/// Basic LAB histogram matching (WASM export)
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_basic_lab(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    // Convert uint8 to float (0-1)
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    basic_lab::color_correct_basic_lab(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        keep_luminosity,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// Basic RGB histogram matching (WASM export)
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_basic_rgb(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    basic_rgb::color_correct_basic_rgb(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// CRA LAB color correction (WASM export)
///
/// Chroma Rotation Averaging in LAB color space. Rotates the AB chroma plane
/// at multiple angles, performs histogram matching at each rotation, then
/// averages the results.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_cra_lab(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    cra_lab::color_correct_cra_lab(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        keep_luminosity,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// Tiled CRA LAB color correction (WASM export)
///
/// CRA with overlapping tile-based processing. Divides the image into blocks
/// with 50% overlap, applies CRA to each block, then blends results using
/// Hamming windows.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     tiled_luminosity: If true, process L channel per-tile before global match
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_tiled_lab(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    tiled_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    tiled_lab::color_correct_tiled_lab(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        tiled_luminosity,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// CRA RGB color correction (WASM export)
///
/// Chroma Rotation Averaging in RGB space. Rotates the RGB cube around the
/// neutral gray axis (1,1,1) using Rodrigues' rotation formula.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     use_perceptual: If true, use perceptual weighting
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_cra_rgb(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    use_perceptual: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    cra_rgb::color_correct_cra_rgb(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        use_perceptual,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// Basic Oklab histogram matching (WASM export)
///
/// Oklab is a perceptually uniform color space with better hue linearity than LAB.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_basic_oklab(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    // Convert uint8 to float (0-1)
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    basic_oklab::color_correct_basic_oklab(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        keep_luminosity,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// CRA Oklab color correction (WASM export)
///
/// Chroma Rotation Averaging in Oklab color space. Rotates the AB chroma plane
/// at multiple angles, performs histogram matching at each rotation, then
/// averages the results. Oklab provides better perceptual uniformity than LAB.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_cra_oklab(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    keep_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    cra_oklab::color_correct_cra_oklab(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        keep_luminosity,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

/// Tiled CRA Oklab color correction (WASM export)
///
/// CRA with overlapping tile-based processing in Oklab color space. Divides the image
/// into blocks with 50% overlap, applies CRA to each block, then blends results using
/// Hamming windows. Combines Oklab's perceptual uniformity with spatial adaptation.
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     tiled_luminosity: If true, process L channel per-tile before global match
///     histogram_mode: 0 = uint8 binned, 1 = f32 endpoint-aligned, 2 = f32 midpoint-aligned
///     histogram_dither_mode: Dither mode for histogram processing (default 4 = Mixed)
///     output_dither_mode: Dither mode for final RGB output (default 2 = Jarvis)
///
/// Returns:
///     Output image as sRGB uint8 (RGBRGB...)
#[wasm_bindgen]
pub fn color_correct_tiled_oklab(
    input_data: &[u8],
    input_width: usize,
    input_height: usize,
    ref_data: &[u8],
    ref_width: usize,
    ref_height: usize,
    tiled_luminosity: bool,
    histogram_mode: u8,
    histogram_dither_mode: u8,
    output_dither_mode: u8,
) -> Vec<u8> {
    let input_srgb: Vec<f32> = input_data.iter().map(|&v| v as f32 / 255.0).collect();
    let ref_srgb: Vec<f32> = ref_data.iter().map(|&v| v as f32 / 255.0).collect();

    tiled_oklab::color_correct_tiled_oklab(
        &input_srgb,
        &ref_srgb,
        input_width,
        input_height,
        ref_width,
        ref_height,
        tiled_luminosity,
        histogram_mode,
        dither_mode_from_u8(histogram_dither_mode),
        dither_mode_from_u8(output_dither_mode),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lab_smoke() {
        // Small 2x2 image
        let input = vec![
            128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150,
        ];
        let reference = vec![
            255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0,
        ];

        // histogram_mode=0 (uint8), histogram_dither_mode=4 (Mixed), output_dither_mode=2 (Jarvis)
        let result = color_correct_basic_lab(&input, 2, 2, &reference, 2, 2, false, 0, 4, 2);
        assert_eq!(result.len(), 12); // 2x2x3 = 12

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_basic_lab(&input, 2, 2, &reference, 2, 2, false, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_basic_lab(&input, 2, 2, &reference, 2, 2, false, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_basic_rgb_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_basic_rgb(&input, 2, 2, &reference, 2, 2, 0, 4, 2);
        assert_eq!(result.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_basic_rgb(&input, 2, 2, &reference, 2, 2, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_basic_rgb(&input, 2, 2, &reference, 2, 2, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_cra_lab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_cra_lab(&input, 2, 2, &reference, 2, 2, false, 0, 4, 2);
        assert_eq!(result.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_cra_lab(&input, 2, 2, &reference, 2, 2, false, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_cra_lab(&input, 2, 2, &reference, 2, 2, false, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_cra_rgb_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_cra_rgb(&input, 2, 2, &reference, 2, 2, false, 0, 4, 2);
        assert_eq!(result.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_cra_rgb(&input, 2, 2, &reference, 2, 2, false, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_cra_rgb(&input, 2, 2, &reference, 2, 2, false, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_basic_oklab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_basic_oklab(&input, 2, 2, &reference, 2, 2, false, 0, 4, 2);
        assert_eq!(result.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_basic_oklab(&input, 2, 2, &reference, 2, 2, false, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_basic_oklab(&input, 2, 2, &reference, 2, 2, false, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_cra_oklab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_cra_oklab(&input, 2, 2, &reference, 2, 2, false, 0, 4, 2);
        assert_eq!(result.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_cra_oklab(&input, 2, 2, &reference, 2, 2, false, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_cra_oklab(&input, 2, 2, &reference, 2, 2, false, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }

    #[test]
    fn test_tiled_oklab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        // Test with tiled luminosity, uint8 histogram
        let result = color_correct_tiled_oklab(&input, 2, 2, &reference, 2, 2, true, 0, 4, 2);
        assert_eq!(result.len(), 12);

        // Test without tiled luminosity (AB only)
        let result_ab = color_correct_tiled_oklab(&input, 2, 2, &reference, 2, 2, false, 0, 4, 2);
        assert_eq!(result_ab.len(), 12);

        // Test with f32 histogram endpoint-aligned
        let result_f32 = color_correct_tiled_oklab(&input, 2, 2, &reference, 2, 2, true, 1, 4, 2);
        assert_eq!(result_f32.len(), 12);

        // Test with f32 histogram midpoint-aligned
        let result_mid = color_correct_tiled_oklab(&input, 2, 2, &reference, 2, 2, true, 2, 4, 2);
        assert_eq!(result_mid.len(), 12);
    }
}
