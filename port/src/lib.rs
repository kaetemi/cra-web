/// CRA (Chroma Rotation Averaging) Color Correction - Rust/WASM Port
///
/// This crate provides WASM-compatible implementations of various color correction
/// algorithms, ported from the original Python scripts.

use wasm_bindgen::prelude::*;

mod basic_lab;
mod basic_rgb;
mod color;
mod cra_lab;
mod cra_rgb;
mod dither;
mod histogram;
mod rotation;
mod tiled_lab;
mod tiling;

// Re-export dithering function for compatibility with existing WASM code
pub use dither::floyd_steinberg_dither;

/// Floyd-Steinberg dithering (WASM export)
/// Matches the existing dither WASM implementation
#[wasm_bindgen]
pub fn floyd_steinberg_dither_wasm(img: Vec<f32>, w: usize, h: usize) -> Vec<u8> {
    dither::floyd_steinberg_dither(&img, w, h)
}

/// Basic LAB histogram matching (WASM export)
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     keep_luminosity: If true, preserve original L channel
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
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
    use_f32_histogram: bool,
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
        use_f32_histogram,
    )
}

/// Basic RGB histogram matching (WASM export)
///
/// Args:
///     input_data: Input image pixels as sRGB uint8 (RGBRGB...)
///     input_width, input_height: Input image dimensions
///     ref_data: Reference image pixels as sRGB uint8 (RGBRGB...)
///     ref_width, ref_height: Reference image dimensions
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
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
    use_f32_histogram: bool,
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
        use_f32_histogram,
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
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
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
    use_f32_histogram: bool,
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
        use_f32_histogram,
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
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
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
    use_f32_histogram: bool,
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
        use_f32_histogram,
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
///     use_f32_histogram: If true, use f32 sort-based histogram matching (no quantization)
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
    use_f32_histogram: bool,
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
        use_f32_histogram,
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

        let result = color_correct_basic_lab(&input, 2, 2, &reference, 2, 2, false, false);
        assert_eq!(result.len(), 12); // 2x2x3 = 12

        // Also test with f32 histogram
        let result_f32 = color_correct_basic_lab(&input, 2, 2, &reference, 2, 2, false, true);
        assert_eq!(result_f32.len(), 12);
    }

    #[test]
    fn test_basic_rgb_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_basic_rgb(&input, 2, 2, &reference, 2, 2, false);
        assert_eq!(result.len(), 12);

        // Also test with f32 histogram
        let result_f32 = color_correct_basic_rgb(&input, 2, 2, &reference, 2, 2, true);
        assert_eq!(result_f32.len(), 12);
    }

    #[test]
    fn test_cra_lab_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_cra_lab(&input, 2, 2, &reference, 2, 2, false, false);
        assert_eq!(result.len(), 12);

        // Also test with f32 histogram
        let result_f32 = color_correct_cra_lab(&input, 2, 2, &reference, 2, 2, false, true);
        assert_eq!(result_f32.len(), 12);
    }

    #[test]
    fn test_cra_rgb_smoke() {
        let input = vec![128, 64, 32, 200, 100, 50, 100, 150, 200, 50, 100, 150];
        let reference = vec![255, 200, 150, 200, 150, 100, 150, 100, 50, 100, 50, 0];

        let result = color_correct_cra_rgb(&input, 2, 2, &reference, 2, 2, false, false);
        assert_eq!(result.len(), 12);

        // Also test with f32 histogram
        let result_f32 = color_correct_cra_rgb(&input, 2, 2, &reference, 2, 2, false, true);
        assert_eq!(result_f32.len(), 12);
    }
}
